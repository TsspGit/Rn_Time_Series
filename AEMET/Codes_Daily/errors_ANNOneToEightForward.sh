#!/bin/bash
for i in {1..8}
do
    python3 errors_NewNN_nForward.py $i > ../logs/errors_ANNv2_"$i"_Fw.log
done
