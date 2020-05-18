#!/bin/bash

SECU_LIST="SZ159949 SH510050 SH510300 SH510500 SH510310 SH512000 SZ159919 SZ159952 SH512760 SH512930" # the default list
TOPDIR_HP=$(realpath ~/wkspaces/HyperPixiu)
STAMP=$(date +%Y%m%dT%H%M%S)
DATAHOME=$(realpath ~/deploy-data/hpdata)
OUTDIR="${DATAHOME}/sim_online"

mkdir -p ${OUTDIR}
if [ -e  ~/deploy-data/hpdata/objectives ]; then
    SECU_LIST=$(cat ~/deploy-data/hpdata/objectives)
fi

cd ${DATAHOME}

FOLDERS=$(ls ${OUTDIR}/ | grep  '\.P.*')
for i in ${FOLDERS}; do
    PID=$(echo $i |sed -e 's/^.*\.P//g');
    mv -vf /tmp/sim_online_${PID}_*.log ${OUTDIR}/${i}/
done

mv ${OUTDIR} ${OUTDIR}.BAK${STAMP}
mkdir -p ${OUTDIR}
# inherit the saved object from last-run
cp -vf ${OUTDIR}.BAK${STAMP}/*.sobj ${OUTDIR}/
nice -n 15 bash -c "tar cfvj ${OUTDIR}.BAK${STAMP}.tar.bz2 ${OUTDIR}.BAK${STAMP} ; rm -rf ${OUTDIR}.BAK${STAMP}" &

cd ${TOPDIR_HP}
if ! [ -d ./out ]; then
    ln -s ${DATAHOME} ./out
fi

for s in ${SECU_LIST}; do
        export SYMBOL="$s"
        RUN_ID="sim_online.${SYMBOL}"
        cp -vf ${TOPDIR_HP}/conf/Trader.json ${OUTDIR}/${RUN_ID}_${STAMP}.json
        SED_STATEMENT="s/^[ \t]*\\\"id\\\".*:.*/      \\\"id\\\":\\\"${RUN_ID}\\\", \/\/ updated at ${STAMP}/g"
        sed -i "${SED_STATEMENT}" ${OUTDIR}/${RUN_ID}_${STAMP}.json
        ./run.sh ./src/launch/sim_online.py -f ${OUTDIR}/${RUN_ID}_${STAMP}.json &
done

