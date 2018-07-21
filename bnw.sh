#!/bin/bash 
for (( i=1; i<=$1; i++ ))
do
	python3 convertToBW.py train/s$i/*.png
	python3 convertToBW.py test/s$i/*.png
done