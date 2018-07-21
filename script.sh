#!/bin/bash 
mkdir train
mkdir test
for (( i=1; i<=$1; i++ ))
do
	path1="./Celebs/s$i"
	mkdir train/s$i
	mkdir test/s$i
	for j in $(seq 1 $2)
	do
		path2="$path1/$j.png"
		cp $path2 train/s$i/$j.png
	done

	let "start = $2 + 1"
	for j in $(seq $start $3)
	do
		path2="$path1/$j.png"
		cp $path2 test/s$i/$j.png
	done
	echo Folder $i copied...
done