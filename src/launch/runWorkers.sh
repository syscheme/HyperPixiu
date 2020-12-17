#!/bin/bash
# usage: ./runWorkers.sh [restart]

WORKER_OF=sinaCrawler
LOG_LEVEL=DEBUG
WORKER_COUNT=1
WORK_AS=

export LC_ALL=en_US.UTF-8

if [ -z ${WORK_AS} ] ; then
    WORK_AS=$(cat /mnt/s/.ssh/id_rsa.pub |cut -d ' ' -f3|cut -d '@' -f1)
fi

if [ -z ${WORK_AS} ] ; then
   echo "quit per WORK_AS not detected, sshfs failed?"
   exit -1
fi

cd ~/wkspaces/
rsync -auv --delete --exclude-from /mnt/w/hpx_template/dist/rsync_excl.txt /mnt/w/hpx_template .

cd ~/wkspaces/hpx_template/src

if [ "restart" == "$1" ]; then
    for ((i=0; i < 5; i++)); do
        for c in ${WORKER_OF}; do
        PID=$(ps aux|grep "celery.*${c}" | grep -v 'grep\|beat\|startWorker.sh'|head -1 |awk '{print $2;}')
        if [ -z ${PID} ]; then break; fi
        kill -INT ${PID};
        done
        sleep 2
    done
fi

for c in ${WORKER_OF}; do
    PID=$(ps aux|grep "celery.*${c}" | grep -v 'grep\|beat\|startWorker.sh'|head -1 |awk '{print $2;}')
    if ! [ -z ${PID} ]; then continue; fi
    Q=${c}
    if [ "sina" == "${Q::4}" ]; then Q="${Q:4}"; fi
    Q=${Q,,} # to lower
    echo "rm -rf /tmp/wkr.${c}.log; dapps/startWorker.sh ${c} -l ${LOG_LEVEL} -c ${WORKER_COUNT} -Q ${Q},celery -n ${WORK_AS}@%h" |at now
done
