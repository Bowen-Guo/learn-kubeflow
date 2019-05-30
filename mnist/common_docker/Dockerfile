ARG BASE_IMAGE_TAG=latest
FROM zzn2/tensorflow-with-conda:$BASE_IMAGE_TAG
RUN pip install --upgrade sklearn
RUN pip install tensorflow
COPY ./src/ /work_dir/src/
COPY ./run_pipeline.sh /work_dir/src/