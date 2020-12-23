#!/bin/bash

MODEL="Cnn1Dx4R2.S1548I4A3"
# TRAINERS="111.44.254.183:48642:ResNet2Xd1.S1548I4A3 111.44.254.183:48642:${MODEL=}"

TRAINERS="111.44.254.183:48642:ResNet21.S1548I4A3"

if [ -e ~/hpx_conf/hpx_settings.sh ]; then source ~/hpx_conf/hpx_settings.sh; fi
if [ -z "${PUBLISH_DIR}" ]; then PUBLISH_DIR="$(realpath ~/wkspace/hpx_publish)" ; fi

STAMP=$(date +%m%d%H%M%S)
for i in ${TRAINERS} ; do
    IP=$(echo ${i} |cut -d ':' -f 1)
    PORT=$(echo ${i} |cut -d ':' -f 2)
    tmp=$(echo ${i} |cut -d ':' -f 3)
    if ! [ -z $tmp ]; then
        MODEL=$tmp
    fi

    echo "collecting result of ${MODEL} from ${IP}:${PORT}"

    SCPREMOTE="root@${IP}"
    SSHREMOTE="root@${IP}"

    if ! [ -z ${PORT} ]; then
      SSHREMOTE="-p ${PORT} ${SCPREMOTE}"
      SCPREMOTE="-P ${PORT} ${SCPREMOTE}"
    fi

    echo "collecting from ${IP}:${PORT}"
    DESTDIR="${PUBLISH_DIR}/${IP}_${PORT}"
    mkdir -p ${DESTDIR}
    ls -lh ${DESTDIR} ;

    latestFile=$(ls ${DESTDIR}/${MODEL}_*.tar.bz2 |sort| tail -1)
    latestMd5=
    if ! [ -z ${latestFile} ] && [ -e ${latestFile} ]; then
        latestMd5=$(md5sum ${latestFile}|cut -d ' ' -f 1)
    fi

    echo "checking remote MD5 on ${IP}:${PORT}"
    removeMD5=$(ssh ${SSHREMOTE} "cat /tmp/${MODEL}.md5" | cut -d ' ' -f 1)
    if [ "${removeMD5}" == "${latestMd5}" ] ; then
        echo "remote MD5 ${removeMD5} matches prev ${latestFile}, ignore it"  | tee -a ${DESTDIR}/CollectTrainer.log
        # rm -vf ${currentFile}  | tee -a ${DESTDIR}/CollectTrainer.log
        continue
    fi

    currentFile=${DESTDIR}/${MODEL}_${STAMP}.tar.bz2

    echo "downloading ${MODEL}.tar.bz2 from ${IP}:${PORT}"
    scp ${SCPREMOTE}:/tmp/${MODEL}.tar.bz2 ${currentFile} 2>&1 | tee -a ${DESTDIR}/CollectTrainer.log
    # if [ $? ]; then exit 1; fi
    newMD5=$(md5sum ${currentFile} |cut -d ' ' -f 1)
    echo "new collection ${currentFile} fetched, md5: ${newMD5}"  | tee -a ${DESTDIR}/CollectTrainer.log

    TAR2CLR=$(ls ${DESTDIR}/${MODEL}_*.tar.bz2 |sort| head -n -5)
    echo "evicting old tarballs: ${TAR2CLR}" | tee -a ${DESTDIR}/CollectTrainer.log
    rm -f ${TAR2CLR} ;
    ls -lh ${DESTDIR} ;

done
