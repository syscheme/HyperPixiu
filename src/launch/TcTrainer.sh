#!/bin/bash

# usually put the following in crontab -e:
# */10 * * * * ~/wkspaces/HyperPixiu/src/launch/TcTrainer.sh 2>&1 | tee -a /tmp/TcTrainer.log &

MODEL="state18x32x4Y4F518x1To3action.resnet50"
CONF="Trainer_RTX2080.json"

PROJLOC=~/wkspaces/HyperPixiu
DATASRC="/mnt/e/AShareSample/"

date

cd ${PROJLOC}
OUTDIR="./out/Trainer"

if ! [ -d ${OUTDIR} ]; then 
    mkdir -vp ${OUTDIR}
fi

PID=$(ps aux|grep 'Trainer.py'|grep ${CONF}| grep -v bash|awk '{print $2;}')
if [ -z "${PID}" ]; then
     if ! [ -e conf/${CONF} ]; then
        cp -vf conf/Trainer.json conf/${CONF}
        echo "${CONF} initialized from conf/Trainer.json"
     fi

    ./run.sh src/dnn/Trainer.py -f conf/${CONF} 2>&1 >/dev/null &
    sleep 1 # to wait the above command starts
    PID=$(ps aux|grep 'Trainer.py'|grep ${CONF}|awk '{print $2;}')
    echo "started new Trainer with PID=${PID}"
fi

echo "current ${OUTDIR} before arch"
ls -lh ${OUTDIR}

FN_BEST="${OUTDIR}/${MODEL}.best.h5"
FN_LAST="${OUTDIR}/${MODEL}_trained-last.h5"

if ! [ -e ${FN_BEST} ] ; then 
    echo "no new ${FN_BEST} generated"
elif ! [ -e ${FN_LAST} ] ; then 
    echo "no new ${FN_LAST} generated"
else
    echo "best generated, patching ${FN_LAST} to ${DATASRC}/${MODEL}.h5"
    cp -vf ${FN_LAST} ${DATASRC}/${MODEL}.h5

    echo "packaging ${MODEL} into /tmp/${MODEL}.tar.bz2"
    
    mkdir -p /tmp/${MODEL}/tb
    rm -vf /tmp/${MODEL}/replayTrain_*.log

    cp -vf ${FN_LAST} /tmp/${MODEL}/
    cp -vf ./conf/${CONF} /tmp/${MODEL}/
    mv -v ${OUTDIR}/tb/* /tmp/${MODEL}/tb/
    rm -rf /tmp/${MODEL}/tb
    
    cp -vf /tmp/Trainer_${PID}_*.log /tmp/${MODEL}/
    cp -vf /tmp/Trainer_${PID}_*.log.*bz2 /tmp/${MODEL}/
    
    cd /tmp/${MODEL}/
    nice tar cvfj /tmp/${MODEL}.tar.bz2~ .
    rm -vf /tmp/${MODEL}.tar.bz2; mv -f /tmp/${MODEL}.tar.bz2~ /tmp/${MODEL}.tar.bz2
    md5sum /tmp/${MODEL}.tar.bz2 | tee /tmp/${MODEL}.md5
    ls -lh /tmp/${MODEL}*.tar.bz2 ;
fi
