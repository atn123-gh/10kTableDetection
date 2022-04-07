#!/bin/bash
python3 train_net.py \
    --dataset_name          10kTable-layout \
    --json_annotation_train ../data/train.json \
    --image_path_train      ../data/ \
    --json_annotation_val   ../data/val.json \
    --image_path_val        ../data/ \
    --config-file           faster_rcnn_R_50_FPN_3x.yml \
    OUTPUT_DIR  ../outputs/ \
    SOLVER.IMS_PER_BATCH 2 