FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt-get update -qq &&\
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxrender1 \
    libxext6 \
    wget \
    unzip \
    git \
    python3-tk \
    &&\
  apt-get clean && rm -rf /var/lib/apt/lists/*

RUN cd /opt &&\
  wget https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip &&\
  unzip protoc-3.3.0-linux-x86_64.zip
ENV PATH_TO_PROTOC /opt

RUN cd /opt &&\
  git clone https://github.com/tensorflow/models.git &&\
  cd /opt/models/research/delf &&\
  ${PATH_TO_PROTOC?}/bin/protoc delf/protos/*.proto --python_out=. &&\
  pip install -e . &&\
  cd /opt/models/research/delf/delf/python/delg &&\
  mkdir parameters && cd parameters &&\
  wget https://storage.googleapis.com/delf/r101delg_gldv2clean_20200914.tar.gz &&\
  tar -xvzf r101delg_gldv2clean_20200914.tar.gz
RUN mkdir parameters &&\
  cd parameters &&\
  wget https://storage.googleapis.com/delf/r101delg_gldv2clean_20200914.tar.gz &&\
  tar -xvzf r101delg_gldv2clean_20200914.tar.gz

RUN cd /opt &&\
  git clone https://github.com/filipradenovic/cnnimageretrieval-pytorch

ENV PYTHONPATH $PYTHONPATH:/opt/models/research:/opt/cnnimageretrieval-pytorch

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py /workspace/
RUN ln -s /opt/models /workspace/models
