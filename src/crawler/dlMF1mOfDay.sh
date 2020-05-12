#!/bin/bash

LIST=$(bzcat ./symbols.txt.bz2)
DATE=$(date +%Y%m%d)

mkdir -p MF${DATE}
cd MF${DATE}

while ! [ -z "${LIST}" ] ; do
    NEXT_LIST=""
    SKIP_AT_456=0
    SZ_TOTAL=0
    SZ_RETRY=0
    for i in ${LIST}; do
        let "SZ_TOTAL+=1"
        FN="${i}_MF1m${DATE}.json"
        if [ -e ${FN} ]; then
             grep -o netamount ${FN} >/dev/null && continue
        fi

        if [ "0" -ne "${SKIP_AT_456}" ]; then
            NEXT_LIST="${NEXT_LIST} ${i}"
            let "SZ_RETRY+=1"
            continue
        fi
        
        echo "fetching MF1m of ${i}"
        RET=$(wget "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssx_ggzj_fszs?sort=time&num=260&page=1&daima=${i}" -O ${FN} 2>&1|grep -o 'awaiting response.*'| grep -o '[0-9]*')
        if [ "200" == "${RET}" ]; then
            echo "downloaded ${FN}, resp ${RET}"
            continue
        fi
        
        echo "downloading ${i} failed, resp ${RET}"
        rm -f ${i}_MF1m${DATE}.json

        if [ "456" == "${RET}" ]; then
            SKIP_AT_456=1
            NEXT_LIST="${NEXT_LIST} ${i}"
            let "SZ_RETRY+=1"
        fi
    done

    if [ $SKIP_AT_456 ]; then
        echo "will retry ${SZ_RETRY} of ${SZ_TOTAL} in 10-min"
        LIST=${NEXT_LIST}
        sleep 600
    fi
done

echo "MF done, zipping to MF${DATE}.tar.bz2"
tar cfvj ../MF${DATE}.tar.bz2 *


