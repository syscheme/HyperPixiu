#!/bin/bash

SECU_LIST=$(grep -o '^S[HZ][0-9]*' ~/deploy-data/hpdata/advisor_objs.txt |sort|uniq)
TOPDIR_HP=~/wkspaces/HyperPixiu
STAMP=$(date +%Y%m%dT%H%M%S)

# sample of crontab
# 50  6-15/2   * *  1-5   ~/tasks/taskAdvisor.sh 2>&1 > /tmp/taskAdvisor.log &
# 0   16       * *  1-5   ps aux|grep 'advisor.py'| awk '{print $2;}' |xargs kill
cd ${TOPDIR_HP}
OUTDIR=./out/advisor
CONF=$(realpath ~/deploy-data/hpdata/Advisor.json)

PID=$(ps aux|grep 'advisor.py'|grep ${CONF} | grep -v 'run.sh' |awk '{print $2;}' )
if ! [ -z ${PID} ]; then
    echo "an existing advisor is running with PID=${PID}, backup its logfiles"
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
    exit 0
fi

PID_LIST=$(ls ${OUTDIR}/*.tcsv |sed 's/^.*advisor_\([0-9]*\).*tcsv/\1/g')
for i in ${PID_LIST}; do
    mv -vf /tmp/advisor_${i}_*.log* ${OUTDIR}/ ; 
done
cp -vf ${CONF} ${OUTDIR}/

# backup and prepare new ${OUTDIR}
mv -vf ${OUTDIR} ${OUTDIR}.BAK${STAMP}
mkdir -p ${OUTDIR}
mv -vf ${OUTDIR}.BAK${STAMP}/*.ss* ${OUTDIR}/ # inherit from previous safestores
rm -rf ${OUTDIR}/*.lock ${OUTDIR}/*.tcsv* ${OUTDIR}/*.log*

echo "backing up to ${OUTDIR}.BAK${STAMP}"
ls -l ${OUTDIR}.BAK${STAMP}/*
echo "new ${OUTDIR}"
ls -l ${OUTDIR}/*

nice -n 15 bash -c "tar cfvj ${OUTDIR}.BAK${STAMP}.tar.bz2 ${OUTDIR}.BAK${STAMP} ; rm -rf ${OUTDIR}.BAK${STAMP}" &

OBJ_LIST="["
for s in ${SECU_LIST}; do
    OBJ_LIST="${OBJ_LIST}\\\"$s\\\","
done
OBJ_LIST="${OBJ_LIST}]"
echo ${OBJ_LIST}

if ! [ -e ${CONF} ]; then
    echo "no ${CONF} exists, duplicated from ${TOPDIR_HP}/conf/Advisor.json"
    cp -vf ${TOPDIR_HP}/conf/Advisor.json ${CONF}
fi

SED_STATEMENT="s/^[ \t]*\\\"objectives\\\".*:.*/   \\\"objectives\\\": ${OBJ_LIST}, \/\/ updated at ${STAMP}/g"
sed -i "${SED_STATEMENT}" ${CONF}

SED_STATEMENT="s/^.*\\\"console\\\".*:.*/   \\\"console\\\": \\\"False\\\", \/\/ updated at ${STAMP}/g"
sed -i "${SED_STATEMENT}" ${CONF}

./run.sh ./src/launch/advisor.py -f ${CONF} 2>&1 >/dev/null &

sleep 1 # to wait the above command starts
PID=$(ps aux|grep 'advisor.py'|grep ${CONF}|awk '{print $2;}')
echo "started advisor with PID=${PID}"
