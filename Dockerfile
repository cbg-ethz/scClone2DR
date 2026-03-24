FROM mcr.microsoft.com/devcontainers/python:3.10

# System-level dependencies needed to build scipy, h5py, and scikit-fda C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    gfortran \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only build first (avoids pulling in large CUDA binaries)
# and then install Pyro which must match the installed PyTorch version
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir pyro-ppl>=1.8.0

# Install the rest of the scientific Python stack
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install the package itself in editable mode so changes in src/ are reflected immediately
WORKDIR /workspaces/scClone2DR
COPY . .
RUN pip install --no-cache-dir -e ".[notebook]"

RUN pip install --no-cache-dir "numpy<2.0.0" scipy pandas matplotlib
