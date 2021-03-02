#!/bin/bash

if [ -e ~/hpx_conf/hpx_settings.sh ]; then source ~/hpx_conf/hpx_settings.sh; fi

if [ -z "${CONF_DIR}" ]; then CONF_DIR="$(realpath ~/hpx_conf)"; fi
if [ -z "${TOPDIR_HP}" ]; then TOPDIR_HP="$(realpath ~/wkspaces/HyperPixiu)" ; fi
if [ -z "${PUBLISH_DIR}" ]; then PUBLISH_DIR="$(realpath ~/wkspace/hpx_publish)" ; fi

source "./sina_funcs.sh"

SYMBOLLIST=$(bzcat ${CONF_DIR}/symbols.txt.bz2|grep -o '^[Ss].[0-9]*' | tr '[:lower:]' '[:upper:]')
SYMBOLLIST="SZ002881 SH600996 SZ002230"

SINA_TODAY=$(date +%Y%m%d)
DATALEN=300

echo "$(date) $0 starts"
cd /tmp

case ${CMD} in
	MF1m)
        if [ "*" == "$1" ] || [ -z "$1" ]; then
            downloadList downloadMF1m
        else
            downloadMF1m $1 ${1}_${CMD}${SINA_TODAY}.json
        fi
		;;

	MF1d)
        if [ "*" == "$1" ] || [ -z "$1" ]; then
            downloadList downloadMF1d
        else
            downloadMF1d $1 ${1}_${CMD}${SINA_TODAY}.json
        fi
		;;

	KL5m)
        if [ "*" == "$1" ] || [ -z "$1" ]; then
            downloadList downloadKL5m
        else
            downloadKL5m $1 ${1}_${CMD}${SINA_TODAY}.json
        fi
		;;

	KL1d)
        if [ "*" == "$1" ] || [ -z "$1" ]; then
            downloadList downloadKL1d
        else
            downloadKL1d $1 ${1}_${CMD}${SINA_TODAY}.json
        fi
		;;

    dlSinaDay.sh)
        DATALEN=300
        downloadList downloadMF1m
        DATALEN=100
        downloadList downloadKL5m

        cd ~/wkspaces/HyperPixiu
        ./run.sh src/crawler/crawlSina.py |bzip2 -9 - > ${PUBLISH_DIR}/act500_${SINA_TODAY}.txt.bz2
		;;

esac
