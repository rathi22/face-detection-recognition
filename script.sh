#!/bin/bash 
mkdir train
mkdir test
for i in {1..40}
do
	path1="./orl_faces/s$i"
	mkdir train/s$i
	mkdir test/s$i
	for j in $(seq 1 $1)
	do
		path2="$path1/$j.png"
		cp $path2 train/s$i/$j.png
	done

	let "start = $1 + 1"
	for j in $(seq $start 10)
	do
		path2="$path1/$j.png"
		cp $path2 test/s$i/$j.png
	done
	echo Folder $i copied...
done