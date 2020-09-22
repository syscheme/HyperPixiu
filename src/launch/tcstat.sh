#!/bin/bash

CONF="replayTrain_U16TfGpu.json"
PID=$(ps aux|grep 'replayTrain'|grep ${CONF}|awk '{print $2;}')

if [ "$1" == "-w" ]; then
    nvidia-smi|head -10
fi

if [ -z ${PID} ]; then
    echo "no replayTrain is running"
    exit 1
fi

grep 'rebuilt\|saved' /tmp/replayTrain_${PID}_*.log
