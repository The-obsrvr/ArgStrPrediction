FROM nvcr.io/nvidia/pytorch:24.03-py3
SHELL ["/bin/bash", "-lc"]

# --- OS deps ---
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      curl pciutils screen && \
    rm -rf /var/lib/apt/lists/*

# --- Upgrade pip toolchain ---
RUN python -m pip install --no-cache-dir -U pip setuptools wheel

# --- Pin CUDA 12.4 PyTorch stack (torch >= 2.6.0) ---
ARG TORCH_VER=2.6.0
ARG TV_VER=0.21.0
ARG TA_VER=2.6.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

RUN python -m pip uninstall -y torch torchvision torchaudio || true && \
    python -m pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} \
      torch==${TORCH_VER} torchvision==${TV_VER} torchaudio==${TA_VER}

RUN python -m pip install --no-cache-dir flair==0.15.1

# --- Install project requirements (includes transformers >= 4.55.0) ---
WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app
