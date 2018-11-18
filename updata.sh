#!/bin/bash

mkdir phat_cuda
cp -r mallocmc_lib phat_cuda
cp -r phat_lib phat_cuda
cp Makefile phat_cuda
cp *.cpp phat_cuda
cp *.h phat_cuda
rename -x -a.cu phat_cuda/*.cpp
scp -P 23333 -r phat_cuda alex@35.240.181.109:~
rm -rf phat_cuda
