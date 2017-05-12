#!/bin/sh
LD_LIBRARY_PATH="/opt/glibc-2.17/lib/:/oppkg/compilers/gcc/4.8.2/lib64/:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/lib64/stubs:/opt/gridware/pkg/compilers/gcc/4.8.2/lib64:/lib64:/usr/lib64:/usr/lib64/nvidia" /opt/glibc-2.17/lib/ld-2.17.so `which python` cnntime.py

