#!/bin/bash
#usage: $0 <remotesrc> [<local-destdir>]
#CMD=${0##*/}
CLEAN_REMOTE=no
REMOTE_PORT=22

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
if [ "" == "$LOCAL_DIR" ]; then LOCAL_DIR=$(realpath .); fi

cd ${LOCAL_DIR}

ssh -p $REMOTE_PORT ${SRC_HOST} "cd ${SRC_DIR} ; md5sum $SRC_FILE" > ./remote_md5.txt
md5sum -c ./remote_md5.txt 2>/dev/null > ${DIFF_FILE}
FILE_LIST_DIFF=$(grep -v OK ${DIFF_FILE} |cut -d ':' -f 1)
FILE_LIST_MATCHED=$(grep OK ${DIFF_FILE} |cut -d ':' -f 1)
rm -rf ${DIFF_FILE}

for f in $FILE_LIST_MATCHED; do
    echo matched $f ;
done

if [ "yes" == "$CLEAN_REMOTE" ] && ! [ -z $FILE_LIST_MATCHED ]; then
    echo -p $REMOTE_PORT ssh ${SRC_HOST} "cd ${SRC_DIR} ; rm -vf $FILE_LIST_MATCHED"
fi

for f in $FILE_LIST_DIFF; do
    scp -P $REMOTE_PORT ${SRC_HOST}:${SRC_DIR}/$f . ;
done
