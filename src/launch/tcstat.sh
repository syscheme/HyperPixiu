#!/bin/bash

CONF="DQNTrainer_U16TfGpu.json"
PID=$(ps aux|grep 'DQNTrainer.py'|grep ${CONF}|awk '{print $2;}')

if [ "$1" == "-w" ]; then
    nvidia-smi|head -10
fi

if [ -z ${PID} ]; then
    echo "no DQNTrainer is running"
    exit 1
fi

grep 'rebuilt\|saved' /tmp/DQNTrainer_${PID}_*.log
