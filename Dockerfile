ARG CUDA_VERSION="12.1.1"
ARG OS="ubuntu22.04"

ARG CUDA_BUILDER_IMAGE="${CUDA_VERSION}-devel-${OS}"
ARG CUDA_RUNTIME_IMAGE="${CUDA_VERSION}-runtime-${OS}"
FROM nvidia/cuda:${CUDA_BUILDER_IMAGE}

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip python3-venv gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

RUN pip install --upgrade pip && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --upgrade pip && pip install deepspeed bitsandbytes gradio jsonlines

RUN mkdir /code/
RUN mkdir /data/

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY . .

CMD bash
# python ./basic_demo/cli_demo_hf.py --from_pretrained THUDM/cogagent-chat-hf --quant 4
# python ./basic_demo/web_demo.py --from_pretrained cogagent-chat --version chat --quant 4
