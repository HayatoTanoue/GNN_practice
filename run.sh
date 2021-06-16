#!/bin/sh

# コンテナイメージ
IMAGE_NAME="gnn:v1"

docker run --rm -it -v $(pwd):/workspace \
    --gpus all -p 8888:8888 ${IMAGE_NAME} \
    jupyter-lab --no-browser --port=8888 --ip=0.0.0.0 \
    --allow-root --NotebookApp.token=""