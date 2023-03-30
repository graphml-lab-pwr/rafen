FROM ubuntu:20.04

ADD . /dynalign
WORKDIR /dynalign

RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get install -y \
    git \
    wget \
    tmux \
    htop \
    vim \
    less \
    gcc \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa && apt update

RUN apt-get install -y --no-install-recommends \
        python3.9 \
        python3.9-dev \
        python3.9-distutils

RUN wget https://bootstrap.pypa.io/get-pip.py &&  \
	 python3.9 get-pip.py && \
	 ln -s /usr/bin/python3.8 /usr/local/bin/python3

RUN apt-get install -y zsh && \
    chsh -s /bin/zsh && \
    zsh -c "$(wget https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"

RUN pip install \
    PyMySQL \
    mlflow \
    awscli \
    boto3 \
    cryptography