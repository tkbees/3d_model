import argparse
import os
import time

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse

from lrm.inferrer import LRMInferrer

app = FastAPI()

# Specify the directory where generated videos will be saved
current_folder = os.path.dirname(os.path.abspath(__file__))

output_directory = os.path.join(current_folder, "lrm/dumps")
image_directory = os.path.join(current_folder, "assets/sample_input")
cache_directory = os.path.join(current_folder, "lrm/.cache")

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)


def generate_video_from_image(image_path: str) -> str:
    # Your implementation to generate video from image and save it
    # This could be your custom function or a call to a machine learning model
    # Save the generated video to the output directory
    image_name = os.path.basename(image_path).split(".")[0]
    generated_video_path = os.path.join(output_directory, f"{image_name}.glb")
    # Your function to generate video and save it to the specified path
    # Example: generate_video(image_path, generated_video_path)
    # Replace the example with your actual logic

    model_name = 'openlrm-small-obj-1.0'
    source_image = image_path
    dump_path = output_directory
    source_size = -1
    render_size = -1
    mesh_size = 384
    export_video = False
    export_mesh = True

    try:
        with LRMInferrer(model_name=model_name, cache_dir=cache_directory) as inferrer:
            inferrer.infer(
                source_image=source_image,
                dump_path=dump_path,
                source_size=source_size,
                render_size=render_size,
                mesh_size=mesh_size,
                export_video=export_video,
                export_mesh=export_mesh,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # shutil.copyfile(image_path, generated_video_path)
    return generated_video_path


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    start_time = time.time()
    # Save the uploaded image to a temporary location
    image_path = os.path.join(image_directory, file.filename)
    image_name = os.path.basename(image_path).split(".")[0]

    with open(image_path, "wb") as image:
        image.write(file.file.read())

    # Call the function to generate video from the image
    try:
        generated_video_path = generate_video_from_image(image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Return the generated video as a file response
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print("Overall Elapsed time: %0.10f seconds." % elapsed_time)
    return FileResponse(generated_video_path, media_type="model/gltf-binary", filename=f"{image_name}.glb")


@app.get("/hello/")
def hello():
    return {"message": "Hello World!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
