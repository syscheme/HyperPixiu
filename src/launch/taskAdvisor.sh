#!/bin/bash

SECU_LIST="SZ159949 SH510050 SH510300 SH510500 SH510310 SH512000 SZ159919 SZ159952 SH512760 SH512930"
TOPDIR_HP=~/wkspaces/HyperPixiu
BAKSTAMP=$(date +%Y%m%dT%H%M%S)

cd ${TOPDIR_HP}
OUTDIR=./out/advisor
CONFFILE=./conf/Advisor.json

PID_LIST=$(ls ${OUTDIR}/*.tcsv |sed 's/^.*advisor_\([0-9]*\).*tcsv/\1/g')
for i in ${PID_LIST}; do
    mv -vf /tmp/advisor_${i}_*.log ${OUTDIR}/ ; 
done
cp -vf ${CONFFILE} ${OUTDIR}/

mv ${OUTDIR} ${OUTDIR}.BAK${BAKSTAMP}
mkdir -p ${OUTDIR}
#?????TODO  cp -vf ${OUTDIR}.BAK${BAKSTAMP}/*.sobj ${OUTDIR}/
nice -n 15 bash -c "tar cfvj ${OUTDIR}.BAK${BAKSTAMP}.tar.bz2 ${OUTDIR}.BAK${BAKSTAMP} ; rm -rf ${OUTDIR}.BAK${BAKSTAMP}" &

OBJ_LIST="["
for s in ${SECU_LIST}; do
    OBJ_LIST="${OBJ_LIST}\\\"$s\\\","
done
OBJ_LIST="${OBJ_LIST}]"
echo ${OBJ_LIST}

SED_STATEMENT="s/^[ \t]*\\\"objectives\\\".*:.*/   \\\"objectives\\\": ${OBJ_LIST}, \/\/ updated at ${BAKSTAMP}/g"
sed -i "${SED_STATEMENT}" ${CONFFILE}

./run.sh ./src/launch/advisor.py -f ${CONFFILE} &
