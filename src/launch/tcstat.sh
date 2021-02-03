#!/bin/bash

CONF="Trainer.json"
# PID=$(ps aux|grep 'Trainer'|grep ${CONF}|awk '{print $2;}')
PID=$(ps aux|grep 'Trainer'|grep python | awk '{print $2;}')

if [ "$1" == "-w" ]; then
    nvidia-smi|head -10
fi

if [ -z ${PID} ]; then
    echo "no replayTrain is running"
    exit 1
fi

grep 'rebuilt\|saved.*from eval' /tmp/Trainer_${PID}_*.log|tail -8
