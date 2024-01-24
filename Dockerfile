

# Use an official CUDA runtime as a parent image
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

USER 0
# Set the working directory in the container
WORKDIR /app

COPY . /app

# Install Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.1.1 with CUDA 11.8 support
RUN pip install torch==2.1.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Copy the current directory contents into the container at /usr/src/app


# Run any necessary commands
RUN pip install --no-cache-dir -r requirements.txt
# Set the default command to execute
# e.g., python your_script.py

EXPOSE 5015

CMD ["uvicorn", "lrm.api.generate_video_api:app", "--host", "0.0.0.0", "--port", "5015"]