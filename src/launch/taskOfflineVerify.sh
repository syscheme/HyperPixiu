#!/bin/bash
# WEIGHT_SRC="/Trainers/111.44.254.183_48642/*.S1548I4A3_*.bz2"
# WEIGHT_SRC="/Trainers/111.44.254.183_48642/Cnn*.S1548I4A3_*.bz2"
# WEIGHT_SRC="/Trainers/111.44.254.183_48642/ResNet2Xd1.*.bz2"
WEIGHT_SRC="/Trainers/111.44.254.183_48642/Cnn*.S1548I4A3_*.bz2"

# SECU_LIST="SZ159949 $(grep '^SH' /mnt/e/AShareSample/ETF/list.txt |awk '{print $1;}')"
# SECU_LIST="SZ159949 SH510050 SH510300 SH510500 SH510310 SH512000 SZ159919 SZ159952 SH512760 SH512930"
SECU_LIST="SZ159949 SH510050 SH510300 SH510500 SZ159919"
# SECU_LIST="SH510050"

STAMP=$(date +%m%dT%H%M%S)
PROJPATH=/root/wkspaces/HP_advisor

rm -rf /tmp/weightToApp
mkdir -p /tmp/weightToApp
cd /tmp/weightToApp
FROM=$(ls ${WEIGHT_SRC} |tail -1)
echo "${STAMP}> extracting weights file from ${FROM}"
tar xfvj ${FROM} --wildcards '*weights.h5'
WEIGHTFILE=$(ls *weights.h5)
MODEL=$(echo "${WEIGHTFILE}" | head --bytes -12)
MODEL0=$(echo $MODEL|cut -d '.' -f 1)
echo y | cp -vf ${WEIGHTFILE} ${PROJPATH}/out/${MODEL}/weights.h5

SED_STATEMENT="s/\\\"brainId\\\".*:.*/\\\"brainId\\\": \\\"${MODEL0}\\\", \/\/ ${STAMP}/g"
sed -i "${SED_STATEMENT}"  ${PROJPATH}/conf/Trader.json
cat ${PROJPATH}/conf/Trader.json

cd ${PROJPATH}

FOLDERS=$(ls ./out/OfflineSimulator|grep 'sim_offline_')
for i in ${FOLDERS}; do
    mv -vf /tmp/${i}_*.log ./out/OfflineSimulator/${i}/; 
done

nice tar cfvj out/sim_offline_BAK$(date +%m%d%H%M).tar.bz2 out/sim_offline

rm -rf out/sim_offline/*
mkdir -p out/sim_offline

echo "${FROM}" > out/sim_offline/modelAsOf.txt
cp -vf ${FROM} out/sim_offline/

echo "verifying model on symbol list: ${SECU_LIST}"

for s in ${SECU_LIST} ; do
        rm -rf ./out/sim_offline.sobj*
        SYMBOL=$s nice ./run.sh ./src/launch/sim_offline.py;
done &
