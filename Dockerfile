FROM nvcr.io/nvidia/pytorch:23.02-py3

RUN useradd -ms /bin/bash myuser

USER myuser
WORKDIR /workspace
