#!/bin/bash

#CONFIGS
SYMBOL_LIST=$(cat /mnt/e/AShareSample/myInterest.txt)
# SYMBOL_LIST=$(cat /mnt/e/AShareSample/300top50.txt)
# SYMBOL_LIST=$(cat /mnt/e/AShareSample/500top50.txt)

# CSVDIR="/mnt/e/csvset2006plus"
CSVDIR="/mnt/e/csvset2006plus/myInterest" # in the case if there is duplicated csv in 300top50 or 500top50

PROJPATH="/root/wkspaces/HP_advisor"

#RUNTIME VARS
# FILEIN_LIST=$(find ${CSVDIR} -name 'S*.csv*')
# BRAINID="Cnn1Dx4R2"
STAMP=$(date +%m%dT%H%M%S)
THISFILE=$(realpath $0)
LOCKFILE="/tmp/$(basename $0).lockfile"

cd ${PROJPATH}

SED_STATEMENT="s/^[ \t]*\\\"source\\\".*:.*/      \\\"source\\\":\\\"$(echo "${CSVDIR}" | sed 's/\//\\\//g')\\\", \/\/ updated by ideal ${STAMP}/g"
sed -i "${SED_STATEMENT}" ./conf/Ideal.json
SED_STATEMENT="s/^[ \t]*\\\"ideal\\\".*:.*/            \\\"ideal\\\":\\\"T+1\\\", \/\/ updated by ideal ${STAMP}/g"
sed -i "${SED_STATEMENT}" ./conf/Ideal.json
# sed -i "s/\\\"brainId\\\".*:.*/\\\"source\\\":\\\"${BRAINID}\\\", \\\/\\\/ updated by ideal/g" ./conf/Ideal.json
cat ./conf/Ideal.json

for SYMB in ${SYMBOL_LIST}; do
    nextFile="no"
    LOGFILE="/tmp/${SYMB}.ideal.log"

    # acquire the handler of ${SYMB}
    while [ "no" == "${nextFile}" ]; do
        if ! ln $THISFILE $LOCKFILE ; then
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
    echo "generating ideal trade of ${SYMB} according ${CSVDIR}..."
    cd ${PROJPATH}
    SYMBOL=${SYMB} nice ./run.sh ./src/launch/sim_offline.py -f ./conf/Ideal.json |tee ${LOGFILE}

done
