#!/usr/bin/env bash
save_model=./models/
pre_model=./models/
logs=./models/log_training.txt
lr=1e-4

CUDA_VISIBLE_DEVICES='' \
nohup python -u train_model.py --model_dir=${save_model} \
                               --pretrained_model=${pre_model} \
                               --learning_rate=${lr} \
                               --level=L1 \
                               --debug=False \
                               --image_size=112 \
                               --batch_size=128 \
                               > ${logs} 2>&1 &
tail -f ${logs}
