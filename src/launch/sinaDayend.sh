#!/bin/bash
# suggested cron line: 0  16  * *  1-5   ~/tasks/sinaDayend.sh 2>&1 > /tmp/sinaDayend.log &
TOPDIR_HP=~/wkspaces/HyperPixiu

SECU_LIST=$(grep -o '^S[HZ][0-9]*' ~/deploy-data/hpdata/advisor_objs.txt |sort|uniq)
LOCKFILE="/tmp/$(basename $0).lockfile"

HOUR=$(date +%H)
HOUR2EXEC=16
if [ ${HOUR} -lt ${HOUR2EXEC} ]; then 
    let "sleep = ${HOUR2EXEC} - ${HOUR}"
    echo "$*" |at now + $sleep hours
    exit 0
fi

SECU_LIST=$(grep -o '^S[HZ][0-9]*' ~/deploy-data/hpdata/advisor_objs.txt |sort|uniq)
STAMP=$(date +%Y%m%dT%H%M%S)

cd ${TOPDIR_HP}
OUTDIR=./out/advisor
CONF=$(realpath ~/deploy-data/hpdata/Advisor.json)

# step 1. kill the running advisor
PID=$(ps aux|grep 'advisor.py'|grep ${CONF} | grep -v 'run.sh' |awk '{print $2;}' )
if ! [ -z ${PID} ]; then
    echo "an existing advisor is running with PID=${PID}, kill it and backup its logfiles"
    kill -9 ${PID}; sleep 1; kill -9 ${PID};

    cp -vf /tmp/advisor_${PID}_*.log ${OUTDIR}/advisor_${PID}.log
    mv -vf /tmp/advisor_${PID}_*.log.*.bz2  ${OUTDIR}/
    for i in ${OUTDIR}/advisor_${PID}_*.log.*.bz2 ; do
        if ! [ -e $i ]; then continue; fi
        BZASOF=$(bzcat $i |head -1|grep -o '^.\{19\}'|sed 's/[- :]*//g')
        mv -vf $i  ${OUTDIR}/advisor_${PID}.${BZASOF}.log.bz2
    done
    for i in ${OUTDIR}/advisor_${PID}*.tcsv.[0-9]*.bz2 ; do
        if ! [ -e $i ]; then continue; fi
        BZASOF=$(stat -c %y $i | sed 's/[- :]*//g' |cut -d '.' -f1)
        mv -vf $i  ${OUTDIR}/advisor_${PID}.${BZASOF}.tcsv.bz2
    done
fi

cp -vf ${CONF} ${OUTDIR}/

# step 2. backup and prepare new ${OUTDIR} like normal
PID_LIST="$(ls ${OUTDIR}/*.tcsv |sed 's/^.*advisor_\([0-9]*\).*tcsv/\1/g')"
if [ -z ${PID_LIST} ]; then
    echo "skip backing up brand new ${OUTDIR}"
else
    echo "backing up to ${OUTDIR}.BAK${STAMP}"

    for i in ${PID_LIST}; do
        mv -vf /tmp/advisor_${i}_*.log* ${OUTDIR}/ ; 
    done

    mv -vf ${OUTDIR} ${OUTDIR}.BAK${STAMP}
    mkdir -p ${OUTDIR}
    mv -vf ${OUTDIR}.BAK${STAMP}/*.ss* ${OUTDIR}/ # inherit from previous safestores
    rm -rf ${OUTDIR}/*.lock ${OUTDIR}/*.tcsv* ${OUTDIR}/*.log*

    for i in ${OUTDIR}.BAK${STAMP}/advisor_*.tcsv.[0-9]*.bz2 ; do
        if ! [ -e $i ]; then continue; fi
        BZASOF=$(stat -c %y $i | sed 's/[- :]*//g' |cut -d '.' -f1)
        BASENAME=$(basename $i |cut -d '.' -f1)
        mv -vf $i  ${OUTDIR}.BAK${STAMP}/${BASENAME}.${BZASOF}.tcsv.bz2
    done

    ls -l ${OUTDIR}.BAK${STAMP}/*
    echo "new ${OUTDIR}"
    ls -l ${OUTDIR}/*

    nice -n 15 bash -c "tar cfvj ${OUTDIR}.BAK${STAMP}.tar.bz2 ${OUTDIR}.BAK${STAMP} ; rm -rf ${OUTDIR}.BAK${STAMP}"
fi

cd ${TOPDIR_HP}

# step 3. process the collected bz2 files of today
extrdir=${TOPDIR_HP}/out/today
rm -rf ${extrdir} ; mkdir -p ${extrdir} ; cd ${extrdir}

TODAY="$(date +%Y%m%dT)"

# 3.1 extract the advisor.BAK${TODAY}T*.tar.ball
FILES="$(ls ${OUTDIR}.BAK${TODAY}T*.tar.bz2)"
for f in ${FILES}; do
    nice tar xfvj ${f} --wildcards '*.tcsv*' --strip 3
    nice bunzip2 *.bz2
done

TCSVLIST="$(ls *.tcsv |sort)"

# 3.2 filter the evmd from advisor.tcsv
rm -rvf evmd ; mkdir -vp evmd

file1st=$(ls -S $TCSVLIST|head -1) # take the biggest file
# already have the list, no need: SECU_LIST="$(grep -o "S[HZ][0-9]\{6\}" ${file1st} | sort |uniq)"
head -30 ${file1st} |grep '^!evmd' | sort |uniq > evmd/hdr.tcsv
evmdlist="$(grep -o '^!evmd[^,]*' evmd/hdr.tcsv |cut -d '!' -f2)"

for s in ${SECU_LIST}; do
    evmdfile="evmd/${s}_evmd${TODAY}.tcsv"
    grep -h ${s} ${TCSVLIST} | sort |uniq > ${evmdfile}
    for et in ${evmdlist}; do
        grep ${et} evmd/hdr.tcsv > evmd/${s}_${et:4}${TODAY}.tcsv
        grep "^${et}" ${evmdfile} >> evmd/${s}_${et:4}${TODAY}.tcsv
    done
done

cd evmd
nice tar cfvj ${TOPDIR_HP}/out/advmd_${TODAY}.tar.bz2 S*.tcsv
cd ${extrdir}
nice tar cfvj ${TOPDIR_HP}/out/adv_${TODAY}.tar.bz2 $TCSVLIST

cd ${TOPDIR_HP}
rm -rf ${extrdir}

# ============================================
echo "test end"
exit 0
