ARG BASE_IMAGE_TAG=nightly-py3
FROM tensorflow/tensorflow:$BASE_IMAGE_TAG
RUN pip install --upgrade sklearn
COPY ./src/ /work_dir/src/
