#!/bin/bash
for i in {1..8}
do
    python3 errors_NewNN_nForward.py $i > ../logs/errors_NNv2_"$i"_Forward.log
done