#!/bin/bash
GPU=$1
SCRIPT=$2
MIN_FREE_MB=${3:-30000}  # require 30GB free before starting

echo "Waiting for GPU $GPU to have ${MIN_FREE_MB}MB free..."
while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
    if [ "$FREE" -ge "$MIN_FREE_MB" ]; then
        echo "GPU $GPU has ${FREE}MB free. Starting."
        break
    fi
    echo "GPU $GPU has ${FREE}MB free. Waiting 60s..."
    sleep 60
done

conda activate parammute && cd ~/ParamMute && bash $SCRIPT 2>&1 | tee ${SCRIPT%.sh}.log
