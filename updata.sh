#!/bin/bash

mkdir matrix_reduction
cp -r mallocmc_lib matrix_reduction
cp -r phat_lib matrix_reduction
cp Makefile matrix_reduction
cp *.cpp matrix_reduction
cp *.h matrix_reduction
rename -x -a.cu matrix_reduction/*.cpp
scp -P 23333 -r matrix_reduction alex@35.240.181.109:~
rm -rf matrix_reduction
