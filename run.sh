#!/usr/bin/env bash

#cat /dev/null > ./run.sh.out



poetry run python run.py fold=0 exp=exp020_lgbm_minmax_scale_obj

<<ARGS
# 2.5 cycles + 3 epochs
exp=exp00_baseline_EfficientNetB0,exp01_baseline_EfficientNetB0,exp21_effv2s_384
exp.model.backbone=efficientnet_b0
exp.training.batch_size=96
exp.input.size=[256,256] \

exec_time=2021-1002-0327 \

exp23_resnet34d_224
exp24_resnet18d_224
exp25_resnet50d_224
ARGS