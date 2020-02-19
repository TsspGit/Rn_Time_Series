#!/bin/bash
for i in {1..8}
do
    python3 errors_NewCNN_nForward.py $i > ../logs/errors_CNNv2_"$i"_Fw.log
done
