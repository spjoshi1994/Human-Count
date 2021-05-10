#!/bin/bash


export GPUID=0
export NET="tinyyolo"
export TRAIN_DIR="./logs"

export TRAIN_DATA_DIR="/home/softnautics/hdd/ShubhamJoshi/data_utils/training
"
if [ $# -eq 0 ]
then
  echo "Usage: ./scripts/train.sh [options]"
  echo " "
  echo "options:"
  echo "-h, --help                show brief help"
  echo "-net                      (squeezeDet|squeezeDet+|vgg16|resnet50)"
  echo "-gpu                      gpu id"
  echo "-train_dir                directory for training logs"
  exit 0
fi

while test $# -gt 0; do
  case "$1" in
      -h|--help)
      echo "Usage: ./scripts/train.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-net                      (squeezeDet|squeezeDet+|vgg16|resnet50)"
      echo "-gpu                      gpu id"
      echo "-train_dir                directory for training logs"
      exit 0
      ;;
    -net)
      export NET="$2"
      shift
      shift
      ;;
    -gpu)
      export GPUID="$2"
      shift
      shift
      ;;
    -train_dir)
      export TRAIN_DIR="$2"
      shift
      shift
      ;;
    *)
      break
      ;;
  esac
done


python ../src/train.py \
  --dataset=KITTI \
  --pretrained_model_path="" \
  --data_path=$TRAIN_DATA_DIR \
  --image_set=train \
  --train_dir="$TRAIN_DIR/train" \
  --net=$NET \
  --summary_step=1000 \
  --checkpoint_step=1000 \
  --max_steps=2000000 \
  --gpu=$GPUID
