#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <stdlib.h>
#include <cstring>
#include <time.h>

__global__ void hello_cuda()
{
    printf("Hello Cuda world \n");
}

__global__ void uniqe_idx_calc_threadIx( int * input)
{
    int tid = threadIdx.x;
    printf("threadIDX: %d, vlaue : %d \n", tid, input[tid]);
}


__global__ void uniqe_gid_calculation( int * input)
{
    int tid = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;
    int gid = tid + offset;

    printf("blockIdx.x : %d, threadIdx.x : %d, gid : %d, value : %d \n",
		blockIdx.x, tid, gid, input[gid]);

}

__global__ void uniqe_gid_calculation_2d( int * data)
{
    int tid = threadIdx.x;
    int block_offset = blockDim.x* blockIdx.x;
    int row_offset = blockDim.x*gridDim.x*blockIdx.y;

    int gid = row_offset+block_offset+tid;

    printf("blockIdx.x : %d, blockIdx.y : %d, threadIdx.x : %d, gid : %d - data : %d \n",
		blockIdx.x, blockIdx.y, tid, gid, data[gid]);
}

__global__ void uniqe_gid_calculation_2d_2d( int * data)
{
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int num_thread_in_block = blockDim.x + blockDim.y;

    int block_offset =  blockIdx.x * num_thread_in_block;

    int num_thread_row = num_thread_in_block*gridDim.x;
    
    int row_offset = num_thread_row*blockIdx.y;

    int gid = row_offset+block_offset+tid;

    printf("blockIdx.x : %d, blockIdx.y : %d, threadIdx.x : %d, gid : %d - data : %d \n",
		blockIdx.x, blockIdx.y, tid, gid, data[gid]);
}

__global__ void mem_trs_test(int *input)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
	printf("tid : %d , gid : %d, value : %d \n",threadIdx.x,gid,input[gid]);

}

__global__ void mem_trs_test2(int *input,int size )
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size)
	    printf("tid : %d , gid : %d, value : %d \n",threadIdx.x,gid,input[gid]);

}

int main()
{
    dim3 block(4);
    dim3 grid( 8);
    // hello_cuda<<<grid,block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();

    int array_size = 16;
    int array_byte_size = sizeof(int) * array_size;
    int h_data[] = {23,9,4,53,65,12,1,33,87,45,23,12,342,56,44,99};

    for (int i = 0; i<array_size; i++ )
    {
        printf("%d ", h_data[i]);
    }
    printf("\n \n");

    int * d_data;

    cudaMalloc( (void **) &d_data, array_byte_size );
    cudaMemcpy( d_data,h_data ,array_byte_size , cudaMemcpyHostToDevice);

    block = 16;
    grid = 1;

    
    uniqe_idx_calc_threadIx<<<grid,block >>>(d_data);
    
    cudaDeviceSynchronize();
    printf("\n \n");
    block = 4;
    grid = dim3(2,2);
    uniqe_gid_calculation<<<grid,block >>>(d_data);


    cudaDeviceSynchronize();
    block = 4;
	grid = dim3(2,2);

	uniqe_gid_calculation_2d << < grid, block >> > (d_data);
	cudaDeviceSynchronize();

    block = dim3(2,2);
	grid = dim3(2,2);
    printf("----------------------------------------------------\n"); 
	uniqe_gid_calculation_2d_2d << < grid, block >> > (d_data);
	cudaDeviceSynchronize();


    printf("----------------------------------------------------\n"); 
    int size = 150;
    int byte_size = size*sizeof(int);

    int * h_input; // host varible 
    h_input = (int *)malloc(byte_size);

    time_t t;
    srand( (unsigned int ) time(&t));
    for (int i = 0; i < size ; i++)
    {
        h_input[i] = (int) (rand() & 0xff);
        printf(" %d ,",h_input[i]);
    }


    printf("\n----------------------------------------------------\n"); 
    int *d_input;
    cudaMalloc((void **) &d_input,byte_size );

    cudaMemcpy(d_input,h_input,byte_size,cudaMemcpyHostToDevice);

    block = 32;
    grid = 5;

    mem_trs_test<<<grid,block>>>(d_input);
    cudaDeviceSynchronize();
    printf("\n----------------------------------------------------\n");
    block = 32;
    grid = 5;

    mem_trs_test2<<<grid,block>>>(d_input,size);
    cudaDeviceSynchronize();


    cudaFree(d_input);
    free(h_input);
    cudaDeviceReset();
    return 0;
}

