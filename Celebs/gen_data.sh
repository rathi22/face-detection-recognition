#!/bin/bash

counter=1
while IFS='' read -r line || [[ -n "$line" ]]; do
	mkdir s$counter/
	python3 datagen.py -n $1 -s \""$line"\" -d s$counter/ 
	((counter++))
done < Celebs.txt
echo Databasing Complete