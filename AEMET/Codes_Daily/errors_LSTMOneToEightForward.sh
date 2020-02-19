#!/bin/bash
for i in {1..8}
do
    python3 errors_LSTM_nForward.py $i > ../logs/errors_LSTMv2_"$i"_Fw.log
done
