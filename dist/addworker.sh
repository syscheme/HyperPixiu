WORKER=$1
WKR_WSPACE=/mnt/data/hpwkspace
WKR_MASTER_HOST=
# the master was created like below
# useradd -d /mnt/data/hpwkspace hpx # the master
# cd /mnt/data/; chown -R hpx:hpx ${WKR_WSPACE}
# tree -dpug ${WKR_WSPACE}
# ├── [drwxr-xr-x hpx      hpx     ]  archived
# │   └── [drwxr-xr-x hpx      hpx     ]  sina
# └── [drwxr-xr-x root     root    ]  users
#     ├── [lrwxrwxrwx root     root    ]  hpx -> master
#     ├── [drwxr-x--- hpx01    hpx     ]  hpx01
#     │   ├── [lrwxrwxrwx hpx01    hpx     ]  archived -> ../../archived
#     │   ├── [drwxrwxr-x hpx01    hpx     ]  hpx_publish
#     ├── [drwxr-xr-x hpx      hpx     ]  master
#     │   ├── [lrwxrwxrwx root     root    ]  archived -> ../../archived
#     │   ├── [drwxrwxrwx hpx      hpx     ]  hpx_publish
#     │   ├── [lrwxrwxrwx root     root    ]  hpx_template -> /home/wkspaces/hpx_template
#     └── [lrwxrwxrwx root     root    ]  root -> master

useradd -b ${WKR_WSPACE}/users -g hpx ${WORKER}
chmod -R g+xr ${WKR_WSPACE}/users/${WORKER}
sudo -u ${WORKER} bash -c "cd; echo | ssh-keygen -q -t rsa -N ''; tar xfvj ${WKR_WSPACE}/users/master/hpx_template/dist/wkr_prof_template.tar.bz2; chown -R ${WORKER}:hpx *; mkdir -m 755 hpx_publish"

# to delete this worker
# userdel -r ${WORKER}
# to refresh wkr_prof_template
# tar cfvj ../../master/wkr_prof_template.tar.bz2 .bashrc .ssh archived hpx_template

# root@local to execute as ${WORKER},
# remote worker homedir should be sshfs-mounted
#    sshfs -o allow_other,default_permissions,sshfs_sync -p <sshport> ${WORKER}@<master-host>:${WKR_WSPACE} /mnt/w
# if the worker is in a PVE LXC, sshfs happens on the host and expose the mounted w to LXC, the sshfs should follow /etc/{subuid,subgid}
#    sshfs -o uid=100000,gid=100000,allow_other,default_permissions,sshfs_sync -p <sshport> ${WORKER}@<master-host>:${WKR_WSPACE} /mnt/w
#    pct set <LXCID> --mp<Num> mp=/mnt/w,/mnt/w
# in the worker os
#    ln -s /mnt/w/users/${WORKER} /mnt/s
# prepare local workspaces
# mkdir  ~/wkspaces; cd ~/wkspaces
# ln -sf /mnt/s/hpx_publish .
# ln -sf /mnt/w/archived .
# rsync -auv --delete --exclude-from /mnt/w/hpx_template/dist/rsync_excl.txt /mnt/w/hpx_template .
# tree -dpug ~/wkspaces
# /root/wkspaces
# └── [drwxr-xr-x root     root    ]  hpx_template
#    ├── [drwxr-xr-x root     root    ]  conf
#    ├── [drwxr-xr-x root     root    ]  dist
#    ├── [drwxr-xr-x root     root    ]  docker
#    ├── [drwxr-xr-x root     root    ]  kits
#    ├── [drwxr-xr-x root     root    ]  sampleconf
#    ├── [drwxr-xr-x root     root    ]  sina
#    │   ├── [drwxr-xr-x root     root    ]  5min
#    │   ├── [drwxr-xr-x root     root    ]  day
#    │   ├── [drwxr-xr-x root     root    ]  rt
#    │   └── [drwxr-xr-x root     root    ]  tick
#    └── [drwxr-xr-x root     root    ]  src
#        ├── [drwxr-xr-x root     root    ]  aaa
#        ├── [drwxr-xr-x root     root    ]  advisors
#        ├── [drwxr-xr-x root     root    ]  ai
#        │   └── [drwxr-xr-x root     root    ]  perspective
#        │       └── [drwxr-xr-x root     root    ]  5mx1w_dx1y
#        ├── [drwxr-xr-x root     root    ]  broker
#        ├── [drwxr-xr-x root     root    ]  crawler
#        ├── [drwxr-xr-x root     root    ]  dapps
#        │   ├── [drwxr-xr-x root     root    ]  sinaCrawler
#        │   └── [drwxr-xr-x root     root    ]  sinaMaster
#        ├── [drwxr-xr-x root     root    ]  event
#        ├── [drwxr-xr-x root     root    ]  hpGym
#        ├── [drwxr-xr-x root     root    ]  launch
#        ├── [drwxr-xr-x root     root    ]  report
#        └── [drwxr-xr-x root     root    ]  vn
# ls -lh ~/wkspaces
# lrwxrwxrwx  1 root root   15 Dec  9 13:09 archived -> /mnt/w/archived
# lrwxrwxrwx  1 root root   18 Dec  9 13:09 hpx_publish -> /mnt/s/hpx_publish
# drwxr-xr-x  9 root root 4.0K Dec  9 12:49 hpx_template
