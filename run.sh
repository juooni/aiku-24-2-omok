#!/bin/bash

# GPU ID를 첫 번째 인자로 받음
GPU_ID=$1

# 특정 GPU만 사용하도록 환경변수 설정
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Python 파일 실행
python main.py 
