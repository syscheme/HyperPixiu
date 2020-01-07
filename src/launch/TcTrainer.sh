#!/bin/bash

MODEL="VGG16d1.S1548I4A3"

PROJLOC="/home/ubuntu/wkspaces/HyperPixiu"

cd ${PROJLOC}
if ! [ -d ./out/${MODEL} ]; then 
    mkdir -p ./out/${MODEL} ; 
fi

PID=$(ps aux|grep 'DQNTrainer.py'|grep U16TfGpu|awk '{print $2;}')
if [ -z ${PID} ]; then
     if ! [ -e conf/DQNTrainer_U16TfGpu.json ]; then
        cp -f conf/DQNTrainer_VGG16d1.json conf/DQNTrainer_U16TfGpu.json
     fi

    ./run.sh src/hpGym/DQNTrainer.py -f conf/DQNTrainer_U16TfGpu.json 2>&1 >/dev/null &
    PID=$(ps aux|grep 'DQNTrainer.py'|grep VGG16d1|awk '{print $2;}')
    echo "started DQNTrainer with PID ${PID}"
fi

OUTDIR="./out/DQNTrainer_${PID}"

if [ -e ${OUTDIR}/VGG16d1.S1548I4A3.weights.h5 ] ; then 
    echo "patching ${OUTDIR} best into ./out/${MODEL}"
    cp -f ${OUTDIR}/VGG16d1.S1548I4A3.best.h5 ./out/${MODEL}/weights.h5

    echo "packaging ${OUTDIR} into /tmp/${MODEL}_${STAMP}.tar.bz2"
    
    mkdir -p /tmp/${MODEL}/tb
    echo "dir ${OUTDIR} before moving"
    ls -lh ${OUTDIR}
    mv ${OUTDIR}/*.h5 /tmp/${MODEL}/
    mv ${OUTDIR}/tb/* /tmp/${MODEL}/tb/
    echo "dir ${OUTDIR} after moving"
    ls -lh ${OUTDIR}

    cp -f /tmp/DQNTrainer_${PID}_*.log /tmp/${MODEL}/
    mv -f /tmp/DQNTrainer_${PID}_*.log.*bz2 /tmp/${MODEL}/
    
    nice tar cvfj /tmp/${MODEL}.tar.bz2~ /tmp/${MODEL}/*
    rm -f /tmp/${MODEL}.tar.bz2; mv -f /tmp/${MODEL}.tar.bz2~ /tmp/${MODEL}.tar.bz2
    ls -lh /tmp/${MODEL}*.tar.bz2 ;
fi

# # keep the most recent 5 tarballs
# STAMP=$(date +%m%d%H%M%S)
# if [ -e /tmp/${MODEL}.tar.bz2 ] ; then 
#     mv -f /tmp/${MODEL}.tar.bz2 /tmp/${MODEL}_${STAMP}.tar.bz2
#     TAR2CLR=$(ls /tmp/${MODEL}_*.tar.bz2 |sort| head -n -5)
#     echo "evicting old tarballs: ${TAR2CLR}"
#     rm -f ${TAR2CLR} ;
#     ls -lh /tmp/${MODEL}*.tar.bz2 ;
# fi


