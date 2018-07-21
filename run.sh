#!/bin/bash
cd Celebs
./gen_data.sh $1
cd ..
./script.sh $1 $2 $3
python3 train.py $1 $2 $3