#!/bin/bash

SRC_DIR=$(realpath `dirname $0`)

SYMBOLLIST=$(bzcat ${SRC_DIR}/symbols.txt.bz2)
# SYMBOLLIST="SZ002881 SH600996 SZ002230"
DATE=$(date +%Y%m%d)
TARGETDIR="$(realpath ~/hpdata)"

#-------------------------------------
RET=200
UA='Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)'

downloadMF1m()
{
    SYMBOL=$1
    FN=$2

    if [ -e ${FN} ]; then
            grep -o netamount ${FN} >/dev/null && echo "MF1m of ${SYMBOL} has already been downloaded" && return
    fi

    echo "fetching MF1m of ${SYMBOL} to ${FN}"
    RET=$(wget --user-agent="${UA}" "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssx_ggzj_fszs?sort=time&num=260&page=1&daima=${SYMBOL}" -O ${FN} 2>&1|grep -o 'awaiting response.*'| grep -o '[0-9]*')
    if [ "200" == "${RET}" ]; then
        echo "downloaded MF1m of ${SYMBOL} as ${FN}, resp ${RET}"
        return
    fi
    
    echo "failed to download MF1m of ${SYMBOL}, resp ${RET}"
    rm -f ${FN}
}

downloadKL5m()
{
    SYMBOL=$1
    FN=$2

    if [ -e ${FN} ]; then
            grep -o volume ${FN} >/dev/null && echo "KL5m of ${SYMBOL} has already been downloaded" && return
    fi

    echo "fetching KL5m of ${SYMBOL} to ${FN}"
    RET=$(wget --user-agent="${UA}" "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=${SYMBOL}&scale=5&datalen=100" -O ${FN} 2>&1|grep -o 'awaiting response.*'| grep -o '[0-9]*')
    if [ "200" == "${RET}" ]; then
        echo "downloaded KL5m of ${SYMBOL} as ${FN}, resp ${RET}"
        return
    fi
    
    echo "failed to download KL5m of ${SYMBOL}, resp ${RET}"
    rm -f ${FN}
}

downloadList()
{
    FUNC_DL=$1
    CATEGORY=$(echo "$FUNC_DL" | sed -e 's/download//g')
    FOLDER="Sina${CATEGORY}_${DATE}"
    
    mkdir -p ${FOLDER}

    NEXT_LIST=${SYMBOLLIST}
    SKIP_AT_456=0
    while ! [ -z "${NEXT_LIST}" ] ; do
        list="${NEXT_LIST}"
        NEXT_LIST=""
        SKIP_AT_456=0
        SZ_TOTAL=0
        SZ_RETRY=0
        for i in ${list}; do
            let "SZ_TOTAL+=1"
            if [ "0" -ne "${SKIP_AT_456}" ]; then
                NEXT_LIST="${NEXT_LIST} ${i}"
                let "SZ_RETRY+=1"
                continue
            fi

            ${FUNC_DL} ${i} "${FOLDER}/${i}_${CATEGORY}${DATE}.json"

            if [ "456" == "${RET}" ]; then
                SKIP_AT_456=1
                NEXT_LIST="${NEXT_LIST} ${i}"
                let "SZ_RETRY+=1"
            fi

            sleep 0.5
        done

        if ! [ "0" == "$SKIP_AT_456" ]; then
            echo "will retry ${SZ_RETRY} of ${SZ_TOTAL} in 10-min"
            sleep 600
        fi

    done

    echo "${CATEGORY} done, zipping to ${FOLDER}.tar.bz2"
    tar cfvj ./${FOLDER}.tar.bz2 ${FOLDER}/*
    if ! [ -e ./${FOLDER}.tar.bz2 ]; then
            return
    fi

    rm -rf ${FOLDER}
    if [ -d "${TARGETDIR}" ]; then
            mv -vf ./${FOLDER}.tar.bz2 "${TARGETDIR}"
            ls -d Sina${CATEGORY}_*| grep -v 'tar.bz2' |sort | head -n -5 | xargs rm -rf 
    fi
}

cd /tmp

downloadList downloadMF1m
downloadList downloadKL5m