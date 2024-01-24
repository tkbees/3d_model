# Copyright (c) 2023, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch.nn as nn

from .encoders.dino_wrapper import DinoWrapper
from .transformer import TriplaneTransformer
from .rendering.synthesizer import TriplaneSynthesizer


class CameraEmbedder(nn.Module):
    """
    Embed camera features to a high-dimensional vector.
    
    Reference:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L27
    """
    def __init__(self, raw_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class LRMGenerator(nn.Module):
    """
    Full model of the large reconstruction model.
    """
    def __init__(self, camera_embed_dim: int, rendering_samples_per_ray: int,
                 transformer_dim: int, transformer_layers: int, transformer_heads: int,
                 triplane_low_res: int, triplane_high_res: int, triplane_dim: int,
                 encoder_freeze: bool = True, encoder_model_name: str = 'facebook/dino-vitb16', encoder_feat_dim: int = 768):
        super().__init__()
        print(f"camera_embed_dim: {camera_embed_dim}\nrendering_samples_per_ray: {rendering_samples_per_ray}\ntransformer_dim: {transformer_dim}\ntransformer_layers: {transformer_layers}\ntransformer_heads: {transformer_heads}\ntriplane_low_res: {triplane_low_res}\ntriplane_high_res: {triplane_high_res}\ntriplane_dim: {triplane_dim}\nencoder_freeze: {encoder_freeze}\nencoder_model_name: {encoder_model_name}\nencoder_feat_dim: {encoder_feat_dim}")
        # attributes
        self.encoder_feat_dim = encoder_feat_dim
        self.camera_embed_dim = camera_embed_dim

        # modules
        self.encoder = DinoWrapper(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
        )
        self.camera_embedder = CameraEmbedder(
            raw_dim=12+4, embed_dim=camera_embed_dim,
        )
        self.transformer = TriplaneTransformer(
            inner_dim=transformer_dim, num_layers=transformer_layers, num_heads=transformer_heads,
            image_feat_dim=encoder_feat_dim,
            camera_embed_dim=camera_embed_dim,
            triplane_low_res=triplane_low_res, triplane_high_res=triplane_high_res, triplane_dim=triplane_dim,
        )
        self.synthesizer = TriplaneSynthesizer(
            triplane_dim=triplane_dim, samples_per_ray=rendering_samples_per_ray,
        )

    def forward_planes(self, image, camera):
        # image: [N, C_img, H_img, W_img]
        # camera: [N, D_cam_raw]
        print(image.shape)
        print(camera.shape)
        assert image.shape[0] == camera.shape[0], "Batch size mismatch for image and camera"
        N = image.shape[0]

        # encode image
        print("Goining into encoder")
        image_feats = self.encoder(image)
        print("Out of encoder")
        print("checking dimensions of image_feats")
        assert image_feats.shape[-1] == self.encoder_feat_dim, \
            f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"
        print("passed")
        # embed camera
        print("Goining into camera_embedder")
        camera_embeddings = self.camera_embedder(camera)
        print("Out of camera_embedder")
        print("checking dimensions of camera_embeddings")
        assert camera_embeddings.shape[-1] == self.camera_embed_dim, \
            f"Feature dimension mismatch: {camera_embeddings.shape[-1]} vs {self.camera_embed_dim}"
        print("passed")
        # transformer generating planes
        print("Goining into transformer")
        print(image_feats.shape)
        print(camera_embeddings.shape)
        planes = self.transformer(image_feats, camera_embeddings)
        print("Out of transformer")
        assert planes.shape[0] == N, "Batch size mismatch for planes"
        assert planes.shape[1] == 3, "Planes should have 3 channels"

        return planes

    def forward(self, image, source_camera, render_cameras, render_size: int):
        # image: [N, C_img, H_img, W_img]
        # source_camera: [N, D_cam_raw]
        # render_cameras: [N, M, D_cam_render]
        # render_size: int
        assert image.shape[0] == source_camera.shape[0], "Batch size mismatch for image and source_camera"
        assert image.shape[0] == render_cameras.shape[0], "Batch size mismatch for image and render_cameras"
        N, M = render_cameras.shape[:2]

        planes = self.forward_planes(image, source_camera)

        # render target views
        render_results = self.synthesizer(planes, render_cameras, render_size)
        assert render_results['images_rgb'].shape[0] == N, "Batch size mismatch for render_results"
        assert render_results['images_rgb'].shape[1] == M, "Number of rendered views should be consistent with render_cameras"

        return {
            'planes': planes,
            **render_results,
        }
