#!/bin/bash

COLLECTION_ROOT="/mnt/t/BigData/deployments-archives/"
DATE_OF_MONDAY=$(date +%Y%m%d -d 'last friday -4 days')
PROJPATH="/mnt/d/workspace.t3600/HyperPixiu"

# DATE_OF_MONDAY="20200817" # for test

DATE_OF_SATDAY=$(date +%Y%m%d -s "${DATE_OF_MONDAY} +5 days") # supposed to be the today of this run
WORK_ROOT="/tmp/SinaWeek.${DATE_OF_MONDAY}"
LOCKFILE="/tmp/$(basename $0).lockfile"
THISFILE=$(realpath $0)
GEN_TAG="${WORK_ROOT}/batch_9999.req"

# section 1 collecting the src data and generates the batch requests
# -------------------------------------------------------------------
while ! [ -e ${GEN_TAG} ]; do
    if ! ln -s $THISFILE $LOCKFILE ; then
        echo "yield for $LOCKFILE"
        sleep 1
        continue
    fi

    if [ -e ${GEN_TAG} ]; then break; fi

    # step 1 collecting the src data
    echo "acquired $LOCKFILE, collecting data between ${DATE_OF_MONDAY} and ${DATE_OF_SATDAY} into ${WORK_ROOT}"
    cd ~

    cd ${WORK_ROOT}

    rm -rfv  ${WORK_ROOT}
    mkdir -p ${WORK_ROOT}
    cd ${WORK_ROOT}

    #part 1. SinaKL??_20200620.tar.bz2, SinaMF??_20200620.tar.bz2 files
    # ./SinaKL5m_20200817.tar.bz2-> .../SinaKL5m_20200817.tar.bz2
    for cat in SinaKL5m SinaMF1m; do
        FILES="$(find ${COLLECTION_ROOT} -name ${cat}*.tar.bz2)"
        for f in ${FILES}; do
            filedate=$(basename ${f}|cut -d '.' -f 1|cut -d '_' -f 2)
            if [ ${filedate} -lt ${DATE_OF_MONDAY} -o ${filedate} -gt ${DATE_OF_SATDAY} ] ; then continue; fi
            tar xfvj ${f}
        done
    done

    for cat in SinaKL1d SinaMF1d; do
        LATEST_1d=""
        if [ -z "$(ls |grep ${cat})" ]; then
            FILES="$(find ${COLLECTION_ROOT} -name ${cat}*.tar.bz2 |sort)"
            for f in ${FILES}; do
                filedate=$(basename ${f}|cut -d '.' -f 1| cut -d '_' -f 2)
                if [ ${filedate} -ge ${DATE_OF_SATDAY} ] ; then
                    LATEST_1d=$f
                    break
                fi
            done

            if ! [ -z ${LATEST_1d} ]; then
                tar xfvj ${LATEST_1d}
            fi
        fi
    done 

    # #part 2. ./advisor_20200817.A300-tc.tar.bz2->.../advisor.BAK20200817T065001.tar.bz2
    # FILES=$(find ${COLLECTION_ROOT} -name advisor.BAK*.tar.bz2)
    # FILES="${FILES}"
    # for f in ${FILES}; do
    #     filedate=$(basename ${f}|cut -d '.' -f 2|cut -d 'T' -f 1|grep -o '[0-9]*')
    #     if [ ${filedate} -lt ${DATE_OF_MONDAY} -o ${filedate} -gt ${DATE_OF_SATDAY} ] ; then continue; fi
    #     hostby=$(basename $(dirname ${f})|cut -d '.' -f 1)
    #     # ln -svf ${f} ./advisor_${filedate}.${hostby}.tar.bz2
    #     extrdir=${WORK_ROOT}/adv_${filedate}.${hostby}
    #     rm -rfv ${extrdir}
    #     mkdir -pv ${extrdir}
    #     cd ${extrdir}
    #     nice tar xfvj ${f} --wildcards '*.tcsv*' --strip 3
    #     nice bunzip2 *.bz2
    #     TCSVLIST="$(ls |sort)"

    #     # ------------------------
    #     # filter the evmd from advisor.tcsv
    #     rm -rvf evmd
    #     mkdir -vp evmd

    #     file1st=$(ls -S $TCSVLIST|head -1) # take the biggest file
    #     symbollist="$(grep -o "S[HZ][0-9]\{6\}" ${file1st} | sort |uniq)"
    #     head -30 ${file1st} |grep '^!evmd' | sort |uniq > evmd/hdr.tcsv
    #     evmdlist="$(grep -o '^!evmd[^,]*' evmd/hdr.tcsv |cut -d '!' -f2)"

    #     for s in ${symbollist}; do
    #         evmdfile="evmd/${s}_evmd${filedate}.tcsv"
    #         grep -h ${s} ${TCSVLIST} | sort |uniq > ${evmdfile}
    #         for et in ${evmdlist}; do
    #             grep ${et} evmd/hdr.tcsv > evmd/${s}_${et:4}_${filedate}.tcsv
    #             grep "^${et}" ${evmdfile} >> evmd/${s}_${et:4}_${filedate}.tcsv
    #         done
    #         # rm -fv ${evmdfile}
    #     done
    #     cd evmd
    #     nice tar cfvj ${extrdir}/../advmd_${filedate}.${hostby}.tar.bz2 S*.tcsv
    #     cd ${extrdir}

    #     # ------------------------

    #     nice tar cfvj ${extrdir}/../advisor_${filedate}.${hostby}.tar.bz2 $TCSVLIST
    #     cd ${WORK_ROOT}
    #     rm -vrf ${extrdir}
    # done


    # step 2 generate the batches according to the symbols
    # TAR4SYMBOLS=$(find ${COLLECTION_ROOT} -name 'SinaKL5m_*.tar.bz2' | sort | tail -1)
    # ALLSYMBOLS=$(tar tvfj ${TAR4SYMBOLS}| grep -o 'S.[0-9]\{6\}'|sort|uniq)
    LASTKL5mDir="$(ls |grep SinaKL5m|sort|tail -1)"
    ALLSYMBOLS=$(ls ${LASTKL5mDir}/ | grep -o 'S.[0-9]\{6\}'|sort|uniq)
    ALLSYMBOLS="${ALLSYMBOLS}"

    SYMBOLS_BATCH_SIZE=20
    SYMBOLS_BATCH=""
    SYMBOLS_C=0
    BATCH_ID=0

    for S in ${ALLSYMBOLS} ; do
        SYMBOLS_BATCH="${SYMBOLS_BATCH},${S}"
        let "SYMBOLS_C+=1"
        if [ ${SYMBOLS_C} -ge ${SYMBOLS_BATCH_SIZE} ]; then
            let "BATCH_ID+=1"
            printf -v BATSTR "%04d" ${BATCH_ID}
            echo "${SYMBOLS_BATCH:1}" > batch_${BATSTR}.req
            SYMBOLS_C=0
            SYMBOLS_BATCH=""
            continue
        fi
    done

    # ${GEN_TAG} is known as the last batch request
    if ! [ -z ${SYMBOLS_BATCH} ] ; then
        echo "${SYMBOLS_BATCH:1}" > ${GEN_TAG}
    else
        echo "" > ${GEN_TAG}
    fi

    rm -f $LOCKFILE # release the locker
    echo "batch requests prepared in ${WORK_ROOT}, released $LOCKFILE"
    break
    
done # while

# section 2 executing tcsvMerger.py for each batch req
# -------------------------------------------------------------------

BATCHREQ_LIST="$(find ${WORK_ROOT} -name 'batch_*.req'|sort)"

for REQ in ${BATCHREQ_LIST}; do
    nextReq="no"
    REQID=$(echo ${REQ} | grep -o 'batch_[0-9]*')
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
    echo "${REQID}> executing: ${CMD}" |tee -a ${LOGFILE}
    ${CMD} |tee -a ${LOGFILE} &
    PID=$(ps aux|grep python|grep "tcsvMerger.py.*${SYMBOLS_BATCH::20}" |awk '{print $2;}')
    echo "batch${REQID} started as PID ${PID}" |tee -a ${LOGFILE}
    wait # till the background process done
    echo "${REQID}> done" |tee -a ${LOGFILE}
    mv -vf /tmp/tcsvMerger_${PID}_*.log* ${WORK_ROOT}/${REQID}/ |tee -a ${LOGFILE}

done
