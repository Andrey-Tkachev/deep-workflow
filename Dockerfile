FROM ubuntu:18.04
MAINTAINER astkachev <astkachev@yandex-team.ru>

ARG VERSION

ENV COMPONENT_NAME rl-manager

WORKDIR /


COPY pysimgrid /pysimgrid
RUN cd /pysimgrid && bash ./get_simgrid.sh

RUN apt-get update \
    && apt-get install -y cmake g++ python3-setuptools \
    && apt-get install -y libboost-context-dev libboost-program-options-dev libboost-filesystem-dev doxygen graphviz-dev \
    && apt-get install -y python3-pip \
    && python3.6 -m pip install --upgrade pip \
    && python3.6 -m pip install numpy networkx cython torch

RUN apt-get update \
    && apt-get install --no-install-recommends -y apt-utils software-properties-common \
    && apt-get install -y python3 python3-dev python-distribute python3-pip git \
    && apt-get install -y wget \
    && pip3 install --upgrade pip

RUN python3.6 -m pip install numpy networkx cython torch
RUN cd /pysimgrid && python3.6 setup.py install --user

COPY *.py /

CMD /bin/bash
# CMD python3.6 gcn_experiment.py