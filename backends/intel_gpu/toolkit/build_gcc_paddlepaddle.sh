#!/bin/bash


source pre_process.sh


build_type="UNKNOWN_BUILD_TYPE"

if [ "$OPT_DEBUG" -eq 1 ] && [ "$OPT_RELEASE" -eq 1 ]; then
build_type="RelWithDebInfo"
elif [ "$OPT_DEBUG" -eq 1 ]; then
build_type="Debug"
else 
build_type="Release"
fi	

#build_type="Debug"
#build_type="Release"
#out_dir="build_asia"
#out_dir="build_clang_nosan"
#out_dir="build_plain_debug"
#out_dir="build_pybindebug"
out_dir="build_gcc_$build_type"

# Artur cmd line
#cmake_cmd="-DWITH_AVX=ON -DWITH_GPU=OFF -DWITH_MKLDNN=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF -DWITH_PROFILER=OFF -DON_INFER=ON -DCMAKE_BUILD_TYPE=Release -DWITH_TESTING=ON -DWITH_INFERENCE_API_TEST=ON -DWITH_NCCL=OFF -DWITH_COVERAGE=OFF -DCMAKE_C_COMPILER="/usr/bin/gcc-5" -DCMAKE_CXX_COMPILER="/usr/bin/g++-5" -DWITH_LITE=OFF -DWITH_PYTHON=ON -DWITH_DISTRIBUTE=ON -DPY_VERSION=3.6 -DWITH_CRYPTO=OFF "
# do bf16 repro Atrtur
#cmake_cmd="-DWITH_AVX=ON -DWITH_GPU=OFF -DWITH_MKLDNN=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF -DWITH_PROFILER=OFF -DON_INFER=ON -DWITH_TESTING=ON -DWITH_INFERENCE_API_TEST=ON -DWITH_NCCL=OFF -DWITH_COVERAGE=OFF  -DWITH_LITE=OFF -DWITH_PYTHON=ON -DWITH_DISTRIBUTE=ON -DPY_VERSION=3.8 -DWITH_CRYPTO=OFF "
#cmake_cmd="-DWITH_CRYPTO=OFF -DWITH_AVX=ON -DWITH_GPU=OFF -DWITH_MKLDNN=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF -DWITH_PROFILER=OFF -DON_INFER=ON -DWITH_TESTING=ON -DWITH_INFERENCE_API_TEST=ON -DWITH_NCCL=OFF -DWITH_COVERAGE=OFF  -DWITH_LITE=OFF -DWITH_PYTHON=ON -DWITH_DISTRIBUTE=ON -DPY_VERSION=3.8 -DWITH_CRYPTO=OFF "
#cmake_cmd="-DPY_VERSION=3.9 -DWITH_GPU=OFF -DWITH_TESTING=ON -DON_INFER=ON -DWITH_AVX=ON -DWITH_MKLDNN=ON -WITH_CRYPTO=OFF -DWITH_CUSTOM_DEVICE=ON -DWITH_CUSTOM_KERNEL=ON"
cmake_cmd="-DPY_VERSION=3.9 -DWITH_GPU=OFF -DWITH_TESTING=ON -DON_INFER=ON -DWITH_AVX=ON -DWITH_MKLDNN=ON -WITH_CRYPTO=OFF -DWITH_CUSTOM_DEVICE=ON -DWITH_CUSTOM_KERNEL=ON"

# Moj orginal
#cmake_cmd=" -DWITH_AVX=ON -DWITH_GPU=OFF -DWITH_MKLDNN=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF -DWITH_PROFILER=OFF -DWITH_NCCL=OFF -DWITH_COVERAGE=OFF -DWITH_LITE=OFF -DWITH_PYTHON=ON -DWITH_DISTRIBUTE=ON -DPY_VERSION=3.8 -DWITH_CRYPTO=OFF"

#cmake_cmd="cmake $cmake_cmd -DSANITIZER_TYPE=Address -DCMAKE_BUILD_TYPE=$build_type .." 
#cmake_cmd="cmake $cmake_cmd -DSANITIZER_TYPE=Thread -DCMAKE_BUILD_TYPE=$build_type .." 
cmake_cmd=" $cmake_cmd  -DCMAKE_BUILD_TYPE=$build_type " 

source post_process.sh






