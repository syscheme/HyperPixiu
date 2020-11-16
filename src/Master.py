# encoding: UTF-8
from Application import *

import sys, os, platform, re
from redlock import Redlock, MultipleRedlockException # redis remote locker https://github.com/SPSCommerce/redlock-py/

from pssh.clients import ParallelSSHClient # yum install openssl-devel cmake; pip install parallel-ssh, https://pypi.org/project/parallel-ssh/

# pip install celery[redis]

hosts = ['localhost', 'localhost']
client = ParallelSSHClient(hosts)

output = client.run_command('uname', return_list=True)
for host_output in output:
    for line in host_output.stdout:
        print(line)

