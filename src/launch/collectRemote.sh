#!/bin/bash
#usage: $0 <remotesrc> [<local-destdir>]
#CMD=${0##*/}
CLEAN_REMOTE=no
REMOTE_PORT=22
DATE_EXPIRE=$(date +%Y%m%d -d 'last sunday -2 weeks')

if [ "-C" == "$1" ]; then 
    CLEAN_REMOTE=yes
    shift
fi

if [ "-p" == "$1" ]; then 
    REMOTE_PORT=$2
    shift
    shift
fi

SRC_HOST=$(echo $1 | cut -d ':' -f 1)
SRC_DIR=$(echo $1 | cut -d ':' -f 2)
SRC_FILE=$(basename $SRC_DIR)
SRC_DIR=$(dirname $SRC_DIR)

DIFF_FILE="/tmp/md5sum.txt"

LOCAL_DIR=$2
if ! [ -d ${LOCAL_DIR} ]; then exit 1; fi
cd ${LOCAL_DIR}
echo "$(date +%Y%m%dT%H%M%S)>> collecting ${SRC_HOST}:${SRC_DIR}" | tee ./collectRemote.log

ssh -p $REMOTE_PORT ${SRC_HOST} "cd ${SRC_DIR} ; md5sum $SRC_FILE" > ./remote_md5.txt
md5sum -c ./remote_md5.txt 2>/dev/null > ${DIFF_FILE}
FILE_LIST_DIFF=$(grep -v OK ${DIFF_FILE} |cut -d ':' -f 1)
FILE_LIST_MATCHED=$(grep OK ${DIFF_FILE} |cut -d ':' -f 1)
FILE_LIST_EXP=""
FILE_LIST_DIFF="$FILE_LIST_DIFF"
rm -rf ${DIFF_FILE}

for f in $FILE_LIST_MATCHED; do
    echo matched $f | tee -a ./collectRemote.log
    dateAsOf=$(echo $f | grep -o '20[0-9]\{6\}')
    if [ "" != "$dateAsOf" ] && [ "$dateAsOf" -lt "$DATE_EXPIRE" ] ; then
        FILE_LIST_EXP="$FILE_LIST_EXP $f"
    fi
done

# echo "FILE_LIST_EXP=$FILE_LIST_EXP"
echo "different files: $FILE_LIST_DIFF"
for f in $FILE_LIST_DIFF; do
    echo "downloading $f"
    scp -P ${REMOTE_PORT} ${SRC_HOST}:${SRC_DIR}/$f . | tee -a ./collectRemote.log
done

if [ "yes" == "$CLEAN_REMOTE" ] && ! [ -z "$FILE_LIST_EXP" ]; then
    ssh -p ${REMOTE_PORT} ${SRC_HOST} "cd ${SRC_DIR} ; rm -vf $FILE_LIST_EXP" | tee -a ./collectRemote.log
fi

