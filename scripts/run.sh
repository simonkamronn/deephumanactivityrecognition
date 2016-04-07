#!/bin/bash -ex
REVISION=$(git rev-parse --short HEAD)
HOST=$1
DEVICE=$2
ssh -q ${HOST} << EOF
    set -ex
    cd rbp-research
    git fetch
    git checkout ${REVISION}
    cd ..
    mkdir -p ${REVISION}
    export REVISION=${REVISION}
    export DEVICE=${DEVICE}
    screen -dmS ${REVISION} bash rbp-research/ex.sh
    sleep 1
    tail -f ${REVISION}/out.txt
EOF