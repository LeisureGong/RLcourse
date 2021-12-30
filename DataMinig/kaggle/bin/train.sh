#!/bin/bash

if [ $# -ne 1 ]
then
  echo "usage: <Settings.json>"
  exit -1
fi

python src/train/gbdt-main.py --setting $1 --cut_str 1-11_13-end --learning_rate 0.08 &
python src/train/gbdt-main.py --setting $1 --cut_str 4-11_13-35 --learning_rate 0.1 &

wait