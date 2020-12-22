#!/bin/bash
RFFolder="/mnt/e/h5_to_h5b/RFrmD4M1X5_2006-2018/ETF"
# RFFolder="/mnt/e/h5_to_h5b/RFrmD4M1X5_2006-2018/500top50"

if [ -e ~/hpx_conf/hpx_settings.sh ]; then source ~/hpx_conf/hpx_settings.sh; fi
TOPDIR_HP=~/wkspaces/HP_advisor

THISFILE=$(realpath $0)
LOCKFILE="/tmp/$(basename $0).lockfile"
FILEIN_LIST=$(find ${RFFolder} -name 'RFrm*_S*.h5'|sort)

for FILE in ${FILEIN_LIST}; do
    nextFile="no"
    LOGFILE="${FILE}b.log"

    # acquire the handler of ${FILE}
    while [ "no" == "${nextFile}" ]; do
        if ! ln $THISFILE $LOCKFILE ; then
            echo "yield for $LOCKFILE"
            sleep 1
            continue
        fi

        echo "acquired $LOCKFILE"
        if [ -e ${LOGFILE} ]; then
            nextFile="yes"
            echo "someone else has already been processing ${FILE}"
        else
            echo "" > ${LOGFILE} # declare to take this file
        fi

        rm -f $LOCKFILE # release the locker
        echo "released $LOCKFILE"
        break
    done

    if [ "no" != "${nextFile}" ]; then
        continue
    fi

    # ok, got the token here
    echo "balancing ${FILE}..."
    cd ${TOPDIR_HP}
    nice ./run.sh src/launch/sim_offline.py -z -b ${FILE} |tee ${LOGFILE}

done
