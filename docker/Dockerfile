FROM ubuntu:22.04

RUN apt update
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    nano \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx unzip \
    curl cron ffmpeg \
    ssh openssh-client \
    vim \
    git

WORKDIR /motrack
ENV PYTHONPATH="/motrack"

RUN pip install uv

# Install project Python environment to /opt/venv so it is not overridden
# by the ${CODE}:/motrack volume mount at runtime.
COPY pyproject.toml uv.lock ./
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync
ENV PATH="/opt/venv/bin:$PATH"

# mmcv prebuilt wheel with CUDA support from the OpenMMLab index.
# Must match the torch+CUDA version installed by uv (torch 2.1 + cu121).
RUN /opt/venv/bin/pip install "mmcv>=2.0.0,<2.2.0" \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html

# Install ultralytics + onnxruntime (optional deps)
RUN /opt/venv/bin/pip install ultralytics onnxruntime-gpu

CMD ["bash"]
