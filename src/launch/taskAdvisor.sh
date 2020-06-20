#!/bin/bash

SECU_LIST=$(grep -o '^S[HZ][0-9]*' ~/deploy-data/hpdata/advisor_objs.txt |sort|uniq)
TOPDIR_HP=~/wkspaces/HyperPixiu
STAMP=$(date +%Y%m%dT%H%M%S)

cd ${TOPDIR_HP}
OUTDIR=./out/advisor
CONF=$(realpath ~/deploy-data/hpdata/Advisor.json)

PID=$(ps aux|grep 'advisor.py'|grep ${CONF}|awk '{print $2;}')
if ! [ -z ${PID} ]; then
    echo "an existing advisor is running with PID=${PID}"
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
