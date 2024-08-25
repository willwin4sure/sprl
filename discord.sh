#!/bin/bash

# If there is a file called discord_nohup_pid.txt, kill the process.
if [ -f discord_nohup_pid.txt ]; then
    kill -9 `cat discord_nohup_pid.txt`
    rm discord_nohup_pid.txt
fi

# Spawn a new process and save the PID to discord_nohup_pid.txt.
# Redirect both streams to discord_nohup.out.
nohup ~/.conda/envs/disc/bin/python ./dopamine/bot.py > discord_nohup.out 2>&1 &
echo $! > discord_nohup_pid.txt