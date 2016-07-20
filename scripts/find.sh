#!/bin/bash
for HOST in tethys theia mnemosyne phoebe rhea themis oceanus
    do
        echo ${HOST}
        ssh -q ${HOST} cuda-smi
    done
