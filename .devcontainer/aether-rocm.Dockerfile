# docker build: used to create a docker image from text file
# t ... : Tag flag, used to name the docker image
# -f .env/DockerFile: The file flag, so we tell the flag where to look for this file
# . : build context, means that the root for all copy commands is in the current directory.
#docker build -t aether-noble-rocm -f .env/Dockerfile .

FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive

# Critical environment variables for CuPy + ROCm 7.1
ENV CUPY_INSTALL_USE_HIP=1
ENV ROCM_HOME=/opt/rocm

# Your specific architecture for the RX series/gfx1201
ENV HCC_AMDGPU_TARGET=gfx1201
ENV GPU_TARGETS=gfx1201 

WORKDIR /workspace

COPY . /workspace

# Install native ROCm 7.1 components, Python 3.14, and build tools 
# directly from Ubuntu archives
# Step 1: Install core compilation binaries, native ROCm core + parallel primitives, and Python 3.14
# Step 1: Install core compilation binaries, native ROCm core + parallel primitives, math modules, and Python 3.14
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ninja-build \
    curl \
    ca-certificates \
    rocm-dev \
    libhipcub-dev \
    librocprim-dev \
    libhipblas-dev \
    libhiprand-dev \
    libhipsparse-dev \
    libhipfft-dev \
    libroctx-dev \
    librocsolver-dev \
    python3.14 \
    python3.14-dev \
    && rm -rf /var/lib/apt/lists/*

# Step 2: Ensure PEP 668 isolation constraints are explicitly unlinked
# This resolves the "error: externally-managed-environment" exception
RUN rm -f /usr/lib/python3.14/EXTERNALLY-MANAGED

# Step 3: Fetch verified installation script and provision pip natively onto Python 3.14
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.14

# Step 6: Replicate the /opt/rocm layout CuPy hardcodes internally
RUN mkdir -p /opt/rocm/include /opt/rocm/bin && \
    # These are directory symlinks — keeps all sibling headers intact
    ln -sf /usr/include/hip          /opt/rocm/include/hip          && \
    ln -sf /usr/include/hiprand      /opt/rocm/include/hiprand      && \
    ln -sf /usr/include/hipcub       /opt/rocm/include/hipcub       && \
    ln -sf /usr/include/rocprim      /opt/rocm/include/rocprim      && \
    ln -sf /usr/include/hipblas      /opt/rocm/include/hipblas      && \
    ln -sf /usr/include/hipfft       /opt/rocm/include/hipfft       && \
    ln -sf /usr/include/hipsparse    /opt/rocm/include/hipsparse    && \
    ln -sf /usr/include/rocsolver    /opt/rocm/include/rocsolver    && \
    ln -sf /usr/include/roctracer    /opt/rocm/include/roctracer    && \
    ln -sf /usr/bin/hipcc            /opt/rocm/bin/hipcc
ENV PATH="/opt/rocm/bin:${PATH}"
# Tell CuPy's probe to also search the subdirectories it won't find on its own
ENV CFLAGS="-I/opt/rocm/include/hipblas \
            -I/opt/rocm/include/hipfft \
            -I/opt/rocm/include/hipsparse \
            -I/opt/rocm/include/rocsolver \
            -I/opt/rocm/include/roctracer"

# Upgrade core build tools inside Python 3.14
RUN pip install --no-cache-dir --upgrade setuptools pip wheel


# Install Aether-ML requirements (installs NumPy, etc.)
RUN pip3 install --no-cache-dir \
    numpy \
    jupyterlab \
    matplotlib 

# Compile CuPy from source for gfx1201
# This ensures that ElementwiseKernels work on your specific AMD GPU
RUN git clone --depth 1 --branch v14.1.1 https://github.com/cupy/cupy.git /tmp/cupy && \
    cd /tmp/cupy && \
    git submodule update --init && \
    CUPY_INSTALL_USE_HIP=1 \
    ROCM_HOME=/opt/rocm \
    HCC_AMDGPU_TARGET=gfx1201 \
    pip install --no-cache-dir . && \
    rm -rf /tmp/cupy

RUN pip install -e .

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]