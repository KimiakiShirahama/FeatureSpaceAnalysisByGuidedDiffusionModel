FROM nvidia/cuda:11.7.1-base-ubuntu22.04
# ENV http_proxy=(Our own setting)
# ENV https_proxy=(Our own settig)
RUN apt-get update && apt-get install -y python3-pip sudo wget git vim libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN ln -s /usr/bin/python3.10 /usr/bin/python
WORKDIR /work
