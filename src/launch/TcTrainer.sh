#!/bin/bash

MODEL="VGG16d1.S1548I4A3"
CONF="DQNTrainer_U16TfGpu.json"
# CONF="DQNTrainer_VGG16d1.json"

PROJLOC=~/wkspaces/HyperPixiu

cd ${PROJLOC}
if ! [ -d ./out/${MODEL} ]; then 
    mkdir -p ./out/${MODEL} ; 
fi

PID=$(ps aux|grep 'DQNTrainer.py'|grep ${CONF}|awk '{print $2;}')
if [ -z ${PID} ]; then
     if ! [ -e conf/${CONF} ]; then
        cp -f conf/DQNTrainer_VGG16d1.json conf/${CONF}
     fi

    ./run.sh src/hpGym/DQNTrainer.py -f conf/${CONF} 2>&1 >/dev/null &
    PID=$(ps aux|grep 'DQNTrainer.py'|grep VGG16d1|awk '{print $2;}')
    echo "started DQNTrainer with PID ${PID}"
fi

OUTDIR="./out/DQNTrainer_${PID}"

if [ -e ${OUTDIR}/${MODEL}.weights.h5 ] ; then 
    echo "patching ${OUTDIR} best into ./out/${MODEL}"
    cp -f ${OUTDIR}/${MODEL}.best.h5 ./out/${MODEL}/weights.h5

    echo "packaging ${OUTDIR} into /tmp/${MODEL}_${STAMP}.tar.bz2"
    
    mkdir -p /tmp/${MODEL}/tb
    echo "dir ${OUTDIR} before moving"
    ls -lh ${OUTDIR}
    cp -f ${OUTDIR}/*.json /tmp/${MODEL}/
    mv ${OUTDIR}/*.h5 /tmp/${MODEL}/
    mv ${OUTDIR}/tb/* /tmp/${MODEL}/tb/
    echo "dir ${OUTDIR} after moving"
    ls -lh ${OUTDIR}

    rm -f /tmp/${MODEL}/DQNTrainer_*.log
    cp -f /tmp/DQNTrainer_${PID}_*.log /tmp/${MODEL}/
    cp -f /tmp/DQNTrainer_${PID}_*.log.*bz2 /tmp/${MODEL}/
    
    cd /tmp/${MODEL}/
    nice tar cvfj /tmp/${MODEL}.tar.bz2~ .
    rm -f /tmp/${MODEL}.tar.bz2; mv -f /tmp/${MODEL}.tar.bz2~ /tmp/${MODEL}.tar.bz2
    md5sum /tmp/${MODEL}.tar.bz2 > /tmp/${MODEL}.md5
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
