#!/bin/bash

HOSTID=$(hostname)
echo "${HOSTID}" |grep syscheme || HOSTID=$(hostname -I)

# step.1 about sync thru OneDrive
# expected in crontab
#   @reboot if [ -e ~/tasks/reboot ]; then ~/tasks/reboot 2>&1 > /tmp/reboot.log; fi &
#   @reboot if [ -e ~/tasks/once ]; then rm -rf /tmp/once ; cp -vf $(realpath ~/tasks/once) /tmp/once; rm -rf ~/tasks/once; /tmp/once 2>&1 > /tmp/once.log ; fi &
#   8 12,17,22 * * 1,2,3,4,5 /usr/bin/onedrive --synchronize 2>&1 > /tmp/oncedrive.txt &
if ! [ -e ~/.config/onedrive/config ]; then
    echo "skip_file = \"~*|.~*|*.tmp\""      > ~/.config/onedrive/config
    echo "no_remote_delete = \"yes\""       >> ~/.config/onedrive/config
    echo "skip_dir = \"*\" # only take the white-list in file sync_list" >> ~/.config/onedrive/config
    echo "log_dir = \"/var/log/onedrive/\"" >> ~/.config/onedrive/config
fi

mkdir -p ~/OneDrive/deployments/${HOSTID}/{hpdata,tasks}
ln -sf ~/OneDrive/deployments/${HOSTID} ~/deploy-data
ln -sf ~/deploy-data/tasks ~/tasks
ln -sf ~/deploy-data/hpdata ~/hpdata
rm -rf ~/wkspaces/HyperPixiu/out ; ln -sf ~/hpdata ~/wkspaces/HyperPixiu/out

mkdir -p /var/log/onedrive/

if ! [ -e ~/.config/onedrive/sync_list ]; then
    echo "deployments/${HOSTID}"  > ~/.config/onedrive/sync_list
fi

onedrive --synchronize --resync --no-remote-delete 2>&1 > /tmp/oncedrive.txt &




