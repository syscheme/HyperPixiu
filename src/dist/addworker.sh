WORKER=$1
# the master was created like below
# useradd -d /mnt/data/hpwkspace hpx # the master
# cd /mnt/data/; chown -R hpx:hpx hpwkspace
# [root@tc2 hpwkspace]# ls -l /mnt/data/hpwkspace
# drwxr-xr-x 2 hpx hpx 16384 Nov 12 12:58 archived
# drwxr-xr-x 3 hpx hpx    22 Nov 12 14:18 master
# drwxrwxr-x 4 hpx hpx    29 Nov 14 14:04 workers
# [root@tc2 hpwkspace]# grep hpx /etc/group
# hpx:x:1000:root

useradd -b /mnt/data/hpwkspace/workers -g hpx ${WORKER}
chmod -R g+xr /mnt/data/hpwkspace/workers/${WORKER}
sudo -u ${WORKER} bash -c "cd; echo | ssh-keygen -q -t rsa -N ''; tar xfvj ../../master/hpx_template/src/dist/wkr_prof_template.tar.bz2; chown -R ${WORKER}:hpx *; mkdir -m 755 to_publish"

# to delete this worker
# userdel -r ${WORKER}
# to refresh wkr_prof_template
# tar cfvj ../../master/wkr_prof_template.tar.bz2 .bashrc .ssh archived hpx_template

# the local of a worker, assuming /mnt/s is sshfs-mounted to the ${WORKER}@master-host
# [root@AHS-X1Y wkspaces]# ls -l ~/wkspaces/*
# lrwxrwxrwx 1 root root 15 Nov 14 15:10 hpx_archived -> /mnt/s/archived
# lrwxrwxrwx 1 root root 17 Nov 14 15:16 hpx_publish -> /mnt/s/to_publish
# -rw-rw-rw- 1 root root 204 Nov 14 15:31 hpx_rsync_excl.txt
# drwxrwxrwx 1 root root 26 Nov 14 12:59 hpx_template <- rsync -auv --delete --exclude-from /mnt/s/hpx_template/src/dist/rsync_excl.txt /mnt/s/hpx_template .

