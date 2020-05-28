#!/bin/bash

SECU_LIST=$(grep -o '^S[HZ][0-9]*' ~/deploy-data/hpdata/trade_objs.txt)
TOPDIR_HP=$(realpath ~/wkspaces/HyperPixiu)
STAMP=$(date +%Y%m%dT%H%M%S)

cd ${TOPDIR_HP}

OUTDIR="./out/sim_online"
CONF=$(realpath ~/deploy-data/hpdata/Trader.json)
PIDLIST=$(ps aux|grep 'sim_online.py'|grep " \-f ${OUTDIR}/"|awk '{print $2;}')
if ! [ -z "${PIDLIST}" ]; then
    echo "existing sim_online is running with PIDLIST=${PIDLIST}, you may kill first"
    exit 0
fi

PID_LIST=$(ls -d ${OUTDIR}/* | grep 'Tdr.P' | sed 's/^.*Tdr.P\([0-9]*\)/\1/g')
for i in ${PID_LIST}; do
    mv -vf /tmp/sim_online_${i}_*.log* ${OUTDIR}/ ; 
done

mv -vf ${OUTDIR} ${OUTDIR}.BAK${STAMP}
mkdir -p ${OUTDIR}
cp -vf ${OUTDIR}.BAK${STAMP}/*.ss ${OUTDIR}  # take the existing safestore
rm -rf ${OUTDIR}.BAK${STAMP}/*.lock
nice -n 15 bash -c "tar cfvj ${OUTDIR}.BAK${STAMP}.tar.bz2 ${OUTDIR}.BAK${STAMP} ; rm -rf ${OUTDIR}.BAK${STAMP}" &

if ! [ -e ${CONF} ]; then
    echo "no ${CONF} exists, duplicated from ${TOPDIR_HP}/conf/Trader.json"
    cp -vf ${TOPDIR_HP}/conf/Trader.json ${CONF}
fi

for s in ${SECU_LIST}; do
        export SYMBOL="$s"
        RUN_ID="sim_online.${SYMBOL}"
        cp -vf ${CONF} ${OUTDIR}/${RUN_ID}_${STAMP}.json
        SED_STATEMENT="s/^[ \t]*\\\"id\\\".*:.*/      \\\"id\\\":\\\"${RUN_ID}\\\", \/\/ updated at ${STAMP}/g"
        sed -i "${SED_STATEMENT}" ${OUTDIR}/${RUN_ID}_${STAMP}.json
        ./run.sh ./src/launch/sim_online.py -f ${OUTDIR}/${RUN_ID}_${STAMP}.json &
done

