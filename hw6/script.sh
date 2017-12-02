#!/bin/bash

# vary N
if [ $2 -eq 1 ]
then
for i in {128..1920..128}
do
$1 $i 4 100
done
fi

# Vary npebs
if [ $2 -eq 2 ]
then
for i in {40..200..20}
do
$1 128 $i 100
done
fi

#vary num_iters
if [ $2 -eq 3 ]
then
for i in {100..1000..100}
do
$1 128 20 $i
done
fi
