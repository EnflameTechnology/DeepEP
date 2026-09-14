#!/bin/bash
for tokens in 8 16 32 48 64 96 128 192 256
do
    for ib_ratio in 40 80 120 200
	do
	    python ../test_low_latency_with_ib.py  --num-processes=8 --num-tokens=96 --hidden=7168 --num-topk=8 --num-experts=256 --total-tokens=${tokens} --imbalance-ratio=${ib_ratio}
        if [ $? -eq 0 ]; then
            echo "tokens ${tokens} ib ratio ${ib_ratio} passed"
        else
            echo "tokens ${tokens} ib ratio ${ib_ratio} failed"
        fi
    done
done
