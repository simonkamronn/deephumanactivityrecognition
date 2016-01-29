#!/bin/bash
for HOST in tethys theia mnemosyne phoebe rhea themis oceanus; do
    echo ${HOST}
    ssh -q sdka@${HOST}.compute.dtu.dk cuda-smi
done