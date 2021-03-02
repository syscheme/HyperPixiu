#!/bin/bash

CONF="Trainer_gpu.json"
# PID=$(ps aux|grep 'Trainer'|grep ${CONF}|awk '{print $2;}')
PSLINE="$(ps aux|grep 'Trainer.py'|grep python)"
echo $PSLINE
PID=$(echo $PSLINE| awk '{print $2;}')

if [ "$1" == "-w" ]; then
    nvidia-smi|head -10
fi

if [ -z ${PID} ]; then
    echo "no trainer is running"
    exit 1
fi

grep 'rebuilt\|saved.*from eval' /tmp/Trainer_${PID}_*.log|tail -8
