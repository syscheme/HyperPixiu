TOPDIR=$(realpath $(dirname $0)/.)
PYTHON=/usr/bin/python3
PROGRAM=$1

export PYTHONPATH=${TOPDIR}:${TOPDIR}/src
export PYTHONIOENCODING=UTF-8
export PYTHONUNBUFFERED=1

${PYTHON} ${PROGRAM} $2 $3 $4 $5 $6 $7 $8 $9 

