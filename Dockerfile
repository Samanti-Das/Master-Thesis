FROM nvidia/cuda:10.1-devel-ubuntu18.04

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.8 python3-pip
RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.8 /usr/bin/python3

RUN apt-get update
RUN apt-get -y install ffmpeg libsm6 libxext6
RUN apt-get -y install git
RUN apt-get -y install screen
RUN apt-get -y install python3.8-dev
RUN apt-get -y install python3-pip
RUN python3 -m pip install --upgrade pip
RUN apt-get -y install cmake

COPY . /app/
WORKDIR /app

RUN pip3 install jupyterlab
RUN pip3 install notebook
RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --no-index torch-scatter==2.0.7 -f https://data.pyg.org/whl/torch-1.7.1+cu101.html
RUN pip3 install --no-index torch-sparse==0.6.9 -f https://data.pyg.org/whl/torch-1.7.1+cu101.html
RUN pip3 install --no-index torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.7.1+cu101.html
RUN pip3 install --no-index torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.7.1+cu101.html
RUN pip3 install torch-geometric
RUN pip3 install tensorboard
RUN pip3 install -U albumentations
RUN pip3 install volumentations-3D


ENV PYTHONPATH "${PYTHONPATH}:/app/src"
EXPOSE 8888
EXPOSE 6006

RUN echo "shell /bin/bash" > ~/.screenrc

ENTRYPOINT ["tail", "-f", "/dev/null"]
