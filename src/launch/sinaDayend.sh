#!/bin/bash
# suggested cron line: 0  16  * *  1-5   ~/tasks/sinaDayend.sh 2>&1 > /tmp/sinaDayend.log &
TOPDIR_HP=~/wkspaces/HyperPixiu
WORKDIR=/mnt/data/hpwkdir

SECU_LIST="$(grep -o '^S[HZ][0-9]*' ~/deploy-data/hpdata/advisor_objs.txt |sort|uniq)"
LOCKFILE="/tmp/$(basename $0).lockfile"

HOUR=$(date +%H)
HOUR2EXEC=16
if [ ${HOUR} -lt ${HOUR2EXEC} ]; then 
    let "sleep = ${HOUR2EXEC} - ${HOUR}"
    echo "$*" |at now + $sleep hours
    exit 0
fi

STAMP=$(date +%Y%m%dT%H%M%S)

cd ${TOPDIR_HP}
DATASRC=./out/advisor
BAKDIR=${WORKDIR}/advisor.BAK${STAMP}

CONF=$(realpath ~/deploy-data/hpdata/Advisor.json)

# step 1. kill the running advisor
PID=$(ps aux|grep 'advisor.py'|grep ${CONF} | grep -v 'run.sh' |awk '{print $2;}' )
if ! [ -z ${PID} ]; then
    echo "an existing advisor is running with PID=${PID}, kill it and backup its logfiles"
    kill -9 ${PID}; sleep 1; kill -9 ${PID};

    cp -vf /tmp/advisor_${PID}_*.log ${DATASRC}/advisor_${PID}.log
    mv -vf /tmp/advisor_${PID}_*.log.*.bz2  ${DATASRC}/

    # renaming the log and tcsv files to better format
    for i in ${DATASRC}/advisor_${PID}_*.log.*.bz2 ; do
        if ! [ -e $i ]; then continue; fi
        BZASOF=$(bzcat $i |head -1|grep -o '^.\{19\}'|sed 's/[- :]*//g')
        mv -vf $i  ${DATASRC}/advisor_${PID}.${BZASOF}.log.bz2
    done
    for i in ${DATASRC}/advisor_${PID}*.tcsv.[0-9]*.bz2 ; do
        if ! [ -e $i ]; then continue; fi
        BZASOF=$(stat -c %y $i | sed 's/[- :]*//g' |cut -d '.' -f1)
        mv -vf $i  ${DATASRC}/advisor_${PID}.${BZASOF}.tcsv.bz2
    done
fi

cp -vf ${CONF} ${DATASRC}/

# step 2. backup and prepare new ${DATASRC} like normal
PID_LIST="$(ls ${DATASRC}/*.tcsv |sed 's/^.*advisor_\([0-9]*\).*tcsv/\1/g')"
if [ -z ${PID_LIST} ]; then
    echo "skip backing up brand new ${DATASRC}"
else
    echo "backing up to ${BAKDIR}"

    for i in ${PID_LIST}; do
        mv -vf /tmp/advisor_${i}_*.log* ${DATASRC}/ ; 
    done

    mv -vf ${DATASRC} ${BAKDIR}
    mkdir -p ${DATASRC}
    mv -vf ${BAKDIR}/*.ss* ${DATASRC}/ # inherit from previous safestores
    rm -rf ${DATASRC}/*.lock ${DATASRC}/*.tcsv* ${DATASRC}/*.log*

    for i in ${BAKDIR}/advisor_*.tcsv.[0-9]*.bz2 ; do
        if ! [ -e $i ]; then continue; fi
        BZASOF=$(stat -c %y $i | sed 's/[- :]*//g' |cut -d '.' -f1)
        BASENAME=$(basename $i |cut -d '.' -f1)
        mv -vf $i ${BAKDIR}/${BASENAME}.${BZASOF}.tcsv.bz2
    done

    ls -l ${BAKDIR}/*
    echo "new ${DATASRC}"
    ls -l ${DATASRC}/*
fi

cd ${TOPDIR_HP}

TODAY="$(date +%Y%m%d)"

if [ -d ${BAKDIR} ]; then
    cd ${BAKDIR}
    BZFILES="$(ls advisor_*.${TODAY}*.tcsv.bz2 |sort)"
    TCSVFILES="$(ls advisor_*.${TODAY}*.tcsv |sort)"
else
    echo "no new collection ${BAKDIR}, checking zipped archieves"
    cd ${TOPDIR_HP}/out
    BAKDIR=""
    BAKLIST="$(ls advisor.BAK*.bz2 | sort -r)"
    for f in ${BAKLIST}; do
        TODAY="$(echo ${f}| sed 's/advisor.BAK\([0-9]*\).*.bz2/\1/g')Z"
        if [ -e "advmd_${TODAY}*.tar.bz2" ] ; then continue; fi

        BAKDIR="${WORKDIR}/advisor.BAK${TODAY}"
        mkdir -p ${BAKDIR}; cd ${BAKDIR}
        tar xfvj ${TOPDIR_HP}/out/${f} --wildcards '*.tcsv*' --strip 3
        
        BZFILES="$(ls advisor_*.tcsv.bz2 |sort)"
        TCSVFILES="$(ls advisor_*.tcsv |sort)"
        break
    done

    if [ -z "${BAKDIR}" ]; then
        echo "no archive associated"
        exit 0
    fi
fi


# step 3. process the collected bz2 files of today
extrdir=${WORKDIR}/today
rm -rf ${extrdir} ; mkdir -p ${extrdir} 
cd ${extrdir}

# 3.1 extract the advisor.BAK${TODAY}T*.tar.ball
for f in ${BZFILES}; do
    bzcat ${BAKDIR}/$f > ./${f::-4}
done
for f in ${TCSVFILES}; do
    ln -sf ${BAKDIR}/$f .
done

TCSVLIST="$(ls *.tcsv |sort)"

# 3.2 filter the evmd from advisor.tcsv
rm -rvf evmd ; mkdir -vp evmd

file1st=$(ls -S $TCSVLIST|head -1) # take the biggest file
# already have the list, no need: SECU_LIST="$(grep -o "S[HZ][0-9]\{6\}" ${file1st} | sort |uniq)"
head -30 ${file1st} |grep '^!evmd' | sort |uniq > evmd/hdr.tcsv
evmdlist="$(grep -o '^!evmd[^,]*' evmd/hdr.tcsv |cut -d '!' -f2)"

for s in ${SECU_LIST}; do
    evmdfile="evmd/${s}_evmd_${TODAY}.tcsv"
    grep -h ${s} ${TCSVLIST} | sort |uniq > ${evmdfile}
    for et in ${evmdlist}; do
        grep ${et} evmd/hdr.tcsv > evmd/${s}_${et:4}_${TODAY}.tcsv
        grep "^${et}" ${evmdfile} >> evmd/${s}_${et:4}_${TODAY}.tcsv
    done
done

cd evmd
nice tar cfvj ${TOPDIR_HP}/out/advmd_${TODAY}.tar.bz2 S*.tcsv
cd ${extrdir}
nice tar cfvj ${TOPDIR_HP}/out/adv_${TODAY}.tar.bz2 $TCSVLIST

cd ${TOPDIR_HP}
rm -rf ${extrdir} ${BAKDIR}

# ============================================
echo "test end"
exit 0
