#!/bin/bash

SRCDataFolder="/mnt/e/AShareSample/SinaWeek.20200629"
PROJPATH="/mnt/d/workspace.t3600/HyperPixiu"
SYMBOLS_BATCH_SIZE=100

DATE_OF_MONDAY=$(date +%Y%m%d -d 'last friday -4 days')
DATE_OF_SATDAY=$(date +%Y%m%d -d 'last friday +1 days') # supposed to be the today of this run
# DATE_OF_MONDAY='20200630'

TAR4SYMBOLS=$(find ${SRCDataFolder} -name SinaKL5m_*.tar.bz2 | sort | tail -1)
ALLSYMBOLS=$(tar tvfj ${TAR4SYMBOLS}| grep -o 'S.[0-9]\{6\}'|sort|uniq)
ALLSYMBOLS="${ALLSYMBOLS}"

SYMBOLS_BATCH=""
SYMBOLS_C=0
BATCH_ID=0
cd ${PROJPATH}
for S in ${ALLSYMBOLS} ; do
    SYMBOLS_BATCH="${SYMBOLS_BATCH},${S}"
    let "SYMBOLS_C+=1"
    if [ ${SYMBOLS_C} -ge ${SYMBOLS_BATCH_SIZE} ]; then
        let "BATCH_ID+=1"
        CMD="./run.sh src/launch/tcsvMerger.py -d ${DATE_OF_MONDAY} -x \"${SYMBOLS_BATCH:1}\" "
        echo "batch_$BATCH_ID> executing: ${CMD}"
        ${CMD}
        SYMBOLS_C=0
        SYMBOLS_BATCH=""
        exit 0
    fi
done
