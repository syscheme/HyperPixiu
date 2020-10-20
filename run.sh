#!/bin/bash

TOPDIR=$(realpath $(dirname $0)/.)
PYTHON=/usr/bin/python3
PROGRAM=$1

CONDA_HOME="/opt/miniconda3"
CONDA_ENV=""

if [ -x "${CONDA_HOME}/bin/conda" ]; then
    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$("${CONDA_HOME}/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "${CONDA_HOME}/etc/profile.d/conda.sh" ]; then
            . "${CONDA_HOME}/etc/profile.d/conda.sh"
        else
            export PATH="${CONDA_HOME}/bin:$PATH"
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<

    conda activate ${CONDA_ENV}
    PYTHON=python
fi

export PYTHONPATH=${TOPDIR}:${TOPDIR}/src
export PYTHONIOENCODING=UTF-8
export PYTHONUNBUFFERED=1

shift
${PYTHON} ${PROGRAM} $*

