#!/bin/bash
# Pod bring-up for RunPod, mirroring the SSH + env-export behaviour of RunPod's
# stock /start.sh (minus the nginx/jupyter bits CI never uses). RunPod injects
# the pod's public key as $PUBLIC_KEY and auto-sets the RUNPOD_* env vars.
set -e

setup_ssh() {
    if [[ $PUBLIC_KEY ]]; then
        echo "Setting up SSH..."
        mkdir -p ~/.ssh
        echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
        chmod 700 -R ~/.ssh
        ssh-keygen -A            # generate any missing host keys
        mkdir -p /run/sshd
        /usr/sbin/sshd           # daemonises; start script continues
    fi
}

# Make RUNPOD_* (used by ppo_train.sh's self-terminate watchdog) and PATH visible
# to SSH sessions, matching RunPod's stock start script.
export_env_vars() {
    printenv | grep -E '^RUNPOD_|^PATH=|^_=' \
        | awk -F = '{ print "export " $1 "=\"" $2 "\"" }' >> /etc/rp_environment
    echo 'source /etc/rp_environment' >> ~/.bashrc
}

setup_ssh
export_env_vars

echo "Pod ready."
sleep infinity
