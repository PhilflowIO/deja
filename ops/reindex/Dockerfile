# Bulk reindex on a machine with a GPU.
#
# Rebuilding the whole index is embedding-bound, and fastembed's default
# onnxruntime is CPU-only. This image swaps in the CUDA build so the same
# `deja index` can run on a workstation-class GPU instead of a laptop CPU.
#
# The CUDA base is 12.x on purpose: the onnxruntime-gpu wheels that
# fastembed-gpu pulls are built against CUDA 12, and a newer driver runs them
# under minor version compatibility. Matching the driver's CUDA release would
# not.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# fastembed and fastembed-gpu are two distributions of the same `fastembed`
# module, differing in which onnxruntime they depend on. Installing deja's
# dependencies normally would drag the CPU one in alongside and the import
# would resolve to whichever landed last. So: pin the dependencies by hand,
# with the GPU variant, and install deja itself without resolution.
#
# fastmcp is deliberately absent — this image indexes, it does not serve, and
# `deja.server` is only imported by `deja serve`.
RUN python3 -m pip install \
      "fastembed-gpu>=0.6.0,<1.0" \
      "sqlite-vec==0.1.8" \
      "platformdirs>=4.0"

COPY . /src
RUN python3 -m pip install --no-deps /src

# The model weights are ~117 MB; mount /cache to keep them across runs.
ENV HF_HOME=/cache/hf \
    FASTEMBED_CACHE_PATH=/cache/fastembed

ENTRYPOINT ["deja"]
CMD ["index"]
