#!/bin/bash

this_dir=$(cd $(dirname $0); pwd)
echo ${this_dir}

if [[ $1 == 'delete' ]];then
    rsync -arzP --delete ${this_dir} --exclude=".git" --exclude=".idea" liujiawei@47.98.204.76:/data/liujiawei/home_backup/gitlab
else
    rsync -arzP ${this_dir} --exclude=".git" --exclude=".idea" liujiawei@47.98.204.76:/data/liujiawei/home_backup/gitlab
fi