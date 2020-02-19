#!/bin/bash
for i in 6
do
    python3 errors_GRU_nForward.py $i > ../logs/errors_GRUv2_"$i"_Fw.log
done
