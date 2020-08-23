#!/bin/bash

COLLECTION_ROOT="/mnt/t/BigData/deployments-archives/"
DATE_OF_MONDAY=$(date +%Y%m%d -d 'last friday -4 days')
DATE_OF_SATDAY=$(date +%Y%m%d -d 'last friday +1 days') # supposed to be the today of this run

PROJPATH="/mnt/d/workspace.t3600/HyperPixiu"
WORK_ROOT="/tmp/SinaWeek.${DATE_OF_MONDAY}"
LOCKFILE="/tmp/$(basename $0).lockfile"
THISFILE=$(realpath $0)

# section 1 collecting the src data and generates the batch requests
# -------------------------------------------------------------------
while ! [ -e ${WORK_ROOT}/batch_999.req ]; do
    if ! ln -s $THISFILE $LOCKFILE ; then
        echo "yield for $LOCKFILE"
        sleep 1
        continue
    fi

    if [ -e ${WORK_ROOT}/batch_999.req ]; then break; fi

    # step 1 collecting the src data
    echo "acquired $LOCKFILE, collecting data beteen ${DATE_OF_MONDAY} and ${DATE_OF_SATDAY} into ${WORK_ROOT}"
    cd ~
    rm -rfv  ${WORK_ROOT}
    mkdir -p ${WORK_ROOT}
    cd ${WORK_ROOT}

    #part 1. SinaKL??_20200620.tar.bz2, SinaMF??_20200620.tar.bz2 files
    # ./SinaKL5m_20200817.tar.bz2-> .../SinaKL5m_20200817.tar.bz2
    FILES=$(find ${COLLECTION_ROOT} -name Sina*.tar.bz2)
    FILES="${FILES}"
    for f in ${FILES}; do
        filedate=$(basename ${f}|cut -d '.' -f 1|cut -d '_' -f 2)
        if [ ${filedate} -lt ${DATE_OF_MONDAY} -o ${filedate} -gt ${DATE_OF_SATDAY} ] ; then continue; fi
        ln -svf ${f} .
    done

    #part 2. ./advisor_20200817.A300-tc.tar.bz2->.../advisor.BAK20200817T065001.tar.bz2
    FILES=$(find ${COLLECTION_ROOT} -name advisor.BAK*.tar.bz2)
    FILES="${FILES}"
    for f in ${FILES}; do
        filedate=$(basename ${f}|cut -d '.' -f 2|cut -d 'T' -f 1|grep -o '[0-9]*')
        if [ ${filedate} -lt ${DATE_OF_MONDAY} -o ${filedate} -gt ${DATE_OF_SATDAY} ] ; then continue; fi
        hostby=$(basename $(dirname ${f})|cut -d '.' -f 1)
        ln -svf ${f} ./advisor_${filedate}.${hostby}.tar.bz2
    done

    # step 2 generate the batches according to the symbols
    TAR4SYMBOLS=$(find . -name 'SinaKL5m_*.tar.bz2' | sort | tail -1)
    ALLSYMBOLS=$(tar tvfj ${TAR4SYMBOLS}| grep -o 'S.[0-9]\{6\}'|sort|uniq)
    ALLSYMBOLS="${ALLSYMBOLS}"

    SYMBOLS_BATCH_SIZE=100
    SYMBOLS_BATCH=""
    SYMBOLS_C=0
    BATCH_ID=0

    for S in ${ALLSYMBOLS} ; do
        SYMBOLS_BATCH="${SYMBOLS_BATCH},${S}"
        let "SYMBOLS_C+=1"
        if [ ${SYMBOLS_C} -ge ${SYMBOLS_BATCH_SIZE} ]; then
            let "BATCH_ID+=1"
            printf -v BATSTR "%03d" ${BATCH_ID}
            echo "${SYMBOLS_BATCH:1}" > batch_${BATSTR}.req
            SYMBOLS_C=0
            SYMBOLS_BATCH=""
            continue
        fi
    done

    # batch_999.req is known as the last batch request
    if ! [ -z ${SYMBOLS_BATCH} ] ; then
        echo "${SYMBOLS_BATCH:1}" > batch_999.req
    else
        echo "" > batch_999.req
    fi

    rm -f $LOCKFILE # release the locker
    echo "batch requests prepared in ${WORK_ROOT}, released $LOCKFILE"
    break
done

# section 2 executing tcsvMerger.py for each batch req
# -------------------------------------------------------------------

BATCHREQ_LIST=$(find ${WORK_ROOT} -name 'batch_*.req'|sort)

for REQ in ${BATCHREQ_LIST}; do
    nextReq="no"
    REQID=$(echo ${REQ} | grep -o "batch_[0-9]*")
    LOGFILE="${WORK_ROOT}/${REQID}.log"

    # acquire the handler of ${FILE}
    while [ "no" == "${nextReq}" ]; do
        if ! ln -s $THISFILE $LOCKFILE ; then
            echo "yield for $LOCKFILE"
            sleep 1
            continue
        fi

        echo "acquired $LOCKFILE"
        if [ -e ${LOGFILE} ]; then
            nextReq="yes"
            echo "someone else has already been processing ${REQ}"
        else
            echo "" > ${LOGFILE} # declare to take this file
        fi

        rm -f $LOCKFILE # release the locker
        echo "released $LOCKFILE"
        break
    done

    if [ "no" != "${nextReq}" ]; then
        continue
    fi

    # ok, got the token here
    SYMBOLS_BATCH=$(cat ${REQ})
    if [ -z ${SYMBOLS_BATCH} ]; then continue; fi

    cd ${PROJPATH}
    rm -rf ${WORK_ROOT}/${REQID}
    mkdir -p ${WORK_ROOT}/${REQID}
    # cp -vf ./conf/Advisor.json ${WORK_ROOT}/${REQID}/merge.json

    CMD="nice ./run.sh src/launch/tcsvMerger.py -s ${WORK_ROOT}/ -o ${WORK_ROOT}/${REQID} -d ${DATE_OF_MONDAY} -x \"${SYMBOLS_BATCH}\" "
    echo "batch_$BATCH_ID> executing: ${CMD}"
    ${CMD} |tee ${LOGFILE}

done
