#!/bin/bash

SECU_LIST="SZ159949 SH510050 SH510300 SH510500 SH510310 SH512000 SZ159919 SZ159952 SH512760 SH512930"
TOPDIR_HP=~/wkspaces/HyperPixiu
BAKSTAMP=$(date +%Y%m%dT%H%M%S)
OUTDIR=./out/OnlineSimulator

cd ${TOPDIR_HP}

FOLDERS=$(ls ./out/OnlineSimulator|grep  '\.P.*')
for i in ${FOLDERS}; do
    PID=$(echo $i |sed -e 's/^.*\.P//g');
    mv -vf /tmp/sim_online_${PID}_*.log ./out/OnlineSimulator/${i}/; 
done

mv ${OUTDIR} ${OUTDIR}.BAK${BAKSTAMP}
mkdir -p ${OUTDIR}
cp -vf ${OUTDIR}.BAK${BAKSTAMP}/*.sobj ${OUTDIR}/
nice -n 15 bash -c "tar cfvj ${OUTDIR}.BAK${BAKSTAMP}.tar.bz2 ${OUTDIR}.BAK${BAKSTAMP} ; rm -rf ${OUTDIR}.BAK${BAKSTAMP}" &

for s in ${SECU_LIST}; do
        export SYMBOL="$s"
        ./run.sh ./src/launch/sim_online.py &
done