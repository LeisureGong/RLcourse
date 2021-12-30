#!/bin/bash

if [ $# -ne 1 ]
then
  echo "usage: <Settings.json>"
  exit -1
fi

setting_json=$1
python src/features_file_main.py --setting $setting_json