FROM nvcr.io/nvidia/pytorch:21.06-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y build-essential autoconf libtool pkg-config python3-dev python3-pip python3-numpy git flex bison libbz2-dev
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add
RUN apt-get install -y software-properties-common
RUN pip install nle
RUN pip install numpy tensorboard gym ray seaborn nevergrad ray[default]
