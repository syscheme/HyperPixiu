#!/bin/bash

HOSTID=$(hostname)
echo "${HOSTID}" |grep syscheme || HOSTID=$(hostname -I)

# step.1 about sync thru OneDrive
if ! [ -e ~/.config/onedrive/config ]; then
    echo "skip_file = \"~*|.~*|*.tmp\""      > ~/.config/onedrive/config
    echo "no_remote_delete = \"yes\""       >> ~/.config/onedrive/config
    echo "skip_dir = \"*\" # only take the white-list in file sync_list" >> ~/.config/onedrive/config
    echo "log_dir = \"/var/log/onedrive/\"" >> ~/.config/onedrive/config
fi

mkdir -p /var/log/onedrive/
mkdir -p ~/OneDrive/deployments/${HOSTID}
if ! [ -e ~/.config/onedrive/sync_list ]; then
    echo "deployments/${HOSTID}"  > ~/.config/onedrive/sync_list
fi

onedrive --synchronize 2>&1 > /tmp/oncedrive.txt &




