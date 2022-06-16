#ifndef CUDA_COMMON_CUH
#define CUDA_COMMON_CUH
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

#define gpuErrChk(ans) { gpuAssert((ans),__FILE__,__LINE__); }

void query_device();



#endif