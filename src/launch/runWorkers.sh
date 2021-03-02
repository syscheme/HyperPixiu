#!/bin/bash
# usage: ./runWorkers.sh [restart]

# WORKER_OF=sinaCrawler
# WORK_AS="hpx53"
# WORKER_COUNT=1
# LOG_LEVEL=DEBUG
# export SSH_CMD="ssh"
# export SSH_HOME="${WORK_AS}@tc2.syscheme.com:~"

if [ -e ~/hpx_conf/hpx_settings.sh ]; then source ~/hpx_conf/hpx_settings.sh; fi

if [ -z "${CONF_DIR}" ]; then CONF_DIR="$(realpath ~/hpx_conf)"; fi
if [ -z "${TOPDIR_HP}" ]; then TOPDIR_HP="$(realpath ~/wkspaces/HyperPixiu)" ; fi
if [ -z "${PUBLISH_DIR}" ]; then PUBLISH_DIR="$(realpath ~/wkspace/hpx_publish)" ; fi

export LC_ALL=en_US.UTF-8

if [ -z ${WORK_AS} ] ; then
    WORK_AS=$(cat /mnt/s/.ssh/id_rsa.pub |cut -d ' ' -f3|cut -d '@' -f1)
fi

if [ -z ${WORK_AS} ] ; then
   echo "quit per WORK_AS not detected, sshfs failed?"
   exit -1
fi

cd ~/wkspaces/
RSYNC_CMD="rsync -auv --delete --exclude-from ./hpx_template/dist/rsync_excl.txt "
if ! [ -z "${SSH_CMD}" ] ; then
    RSYNC_CMD="${RSYNC_CMD} -e \"$SSH_CMD\" ${SSH_HOME}/hpx_template/ ./hpx_template"
else
    RSYNC_CMD="${RSYNC_CMD} /mnt/w/hpx_template/* ./hpx_template"
fi
bash -c "${RSYNC_CMD}"

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
