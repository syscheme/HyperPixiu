#!/bin/bash

SYMBOL_LIST=$(ls /mnt/3T/BigData/AShareKL1m/A2013-2019/2019/ | cut -d '-' -f1)
TOPDIR_HP="/mnt/3T/tmpwks/HyperPixiu"
STAMP=$(date +%Y%m%dT%H%M%S)

CSVDIR="/mnt/3T/BigData/AShareKL1m/A2013-2019" # in the case if there is duplicated csv in 300top50 or 500top50
THISFILE=$(realpath $0)
CONF_FILE="./conf/FutureDays.json"
LOCKFILE="/tmp/$(basename $THISFILE).lockfile"

cd ${TOPDIR_HP}

SED_STATEMENT="s/^[ \t]*\\\"source\\\".*:.*/      \\\"source\\\":\\\"$(echo "${CSVDIR}" | sed 's/\//\\\//g')\\\", \/\/ updated by $(basename $THISFILE) @${STAMP}/g"
sed -i "${SED_STATEMENT}" ${CONF_FILE}
SED_STATEMENT="s/^[ \t]*\\\"ideal\\\".*:.*/            \\\"ideal\\\":\\\"FuturePrice\\\", \/\/ updated by $(basename $THISFILE) @${STAMP}/g"
sed -i "${SED_STATEMENT}" ${CONF_FILE}

cat ${CONF_FILE}

for SYMB in ${SYMBOL_LIST}; do
    nextFile="no"
    LOGFILE="./out/sim_offline/${SYMB}.fd.log"

    # acquire the handler of ${SYMB}
    while [ "no" == "${nextFile}" ]; do
        if ! ln -s $THISFILE $LOCKFILE ; then
            echo "yield for $LOCKFILE"
            sleep 1
            continue
        fi

        echo "acquired $LOCKFILE"
        if [ -e ${LOGFILE} ]; then
            nextFile="yes"
            echo "someone else has already been processing ${SYMB}"
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
    echo "generating future prices of ${SYMB} according ${CSVDIR}..."
    cd ${TOPDIR_HP}
    SYMBOL=${SYMB} nice ./run.sh ./src/launch/sim_offline.py -f ${CONF_FILE} |tee ${LOGFILE}

done
