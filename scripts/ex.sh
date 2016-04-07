#!/bin/bash -ex
source ~/.research
cd ${REVISION}
THEANO_FLAGS=device=${DEVICE} python -u ../rbp-research/rnn/ex.py ../unseen_train.txt ../unseen_val.txt ../training-data/ &> out.txt
cd ..
s3cmd put -r ${REVISION} s3://rbp-research/ts/results/
RESULT=$(grep epoch ${REVISION}/out.txt)
HOST=$(hostname)
curl -v --data-urlencode "text=${RESULT}" "https://slack.com/api/chat.postMessage