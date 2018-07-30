env "PYTHONPATH=/home/huishao/workspace/HyperPixiu:/home/huishao/workspace/HyperPixiu/kits/vnpy" "PYTHONIOENCODING=UTF-8" "PYTHONUNBUFFERED=1" /usr/bin/python2.7 kits/vnpy/vnpy/api/huobi/testmd.py |tee >(bzip2 -c -9 > eos1min_`date +%Y%m%dT%H%M%S`.log.bz2)

