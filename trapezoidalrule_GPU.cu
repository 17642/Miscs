#%%writefile ks.cpp
%%cuda -c "--gpu-architecture sm_75 -Xcompiler=-fopenmp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<stdlib.h>
#include<string.h>
#include<chrono>
#include<iostream>
#include<omp.h>
#include<random>
#include<cmath>


using namespace std;

#define USE_ATOMIC_SUM
//#define USE_CPU_REDUCTION
//#define USE_GPU_GLOBAL_REDUCTION
//#define USE_GPU_SHARED_REDUCTION

#define NUM_THREADS 1024
#define TFUNCTION(x) (x*24-x*x*x)

//__shared__ float sdata;

__device__ __forceinline__ float Target_Function(float x){
  return  TFUNCTION(x);
}

__device__ void calc_with_trapezodial(float x_start, float delta_x, int cuts, float* sdata){
  //extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int idx = blockIdx.x*blockDim.x+tid;

  if(idx<cuts){
    sdata[tid] = Target_Function(x_start+delta_x*idx);
  }

  if(tid==blockDim.x-1 && idx<cuts){
    sdata[blockDim.x] = Target_Function(x_start+delta_x*(idx+1));
  }

  __syncthreads();

}

__device__ void trapezoidal_calc(float halfDelta_x, int cuts, float* sdata){
  int tid = threadIdx.x;
  int idx = blockIdx.x*blockDim.x+tid;

  //extern __shared__ float sdata[];

  float* sdd = sdata;
  float* rdd = sdata+blockDim.x+1;

  if(idx<cuts){
    rdd[tid] = (sdd[tid]+sdd[tid+1])*halfDelta_x;
  }

  __syncthreads();
}

__device__ void reduction_block(float* blksum, float* sdata){
  int tid = threadIdx.x;
  int idx = blockIdx.x*blockDim.x+tid;

  //extern __shared__ float sdata[];
  float* rdd = sdata+blockDim.x+1;

  for(int tli = blockDim.x/2;tli>0;tli>>=1){
    if(tid<tli){
      rdd[tid] = rdd[tid]+rdd[tid+tli];
    }
    __syncthreads();
  }

  if(!tid)blksum[blockIdx.x] = rdd[tid];

}

__global__ void CalcEverything(float x_start, float delta_x, int cuts, float halfDelta_x, float* blksum, float* gpu_sum){
  extern __shared__ float sdata[]; 

  calc_with_trapezodial(x_start, delta_x, cuts, sdata);
  trapezoidal_calc(halfDelta_x, cuts, sdata);
  reduction_block(blksum, sdata);  

  #ifdef USE_ATOMIC_SUM
  if(!threadIdx.x)atomicAdd(gpu_sum, blksum[blockIdx.x]);
  #endif


}

#ifdef USE_GPU_SHARED_REDUCTION
__global__ void SecondaryReduction(int num, float* datas){
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x*blockDim.x+tid;

    if(idx<num){
        sdata[tid] = datas[idx];
    }else{
        sdata[tid] = 0;
    }
    __syncthreads();

    for(int tli = blockDim.x/2;tli>0;tli>>=1){
        if(tid<tli){
            sdata[tid] = sdata[tid]+sdata[tid+tli];
        }
        __syncthreads();  
    }
    if(tid==0)datas[blockIdx.x] = sdata[tid];
}
#endif


float calcByGPU(float start, float end, int cuts){
  float step = (end-start)/cuts;
  float* block_sum;
  float* gpu_sum;
  gpu_sum = new float;
  float* dgpusum;
  cudaMalloc(&dgpusum, sizeof(float));

  float sum = 0;

  int grids = (cuts + NUM_THREADS - 1) / NUM_THREADS;
  int threads = NUM_THREADS;
  int getx = threads+1;
  cudaMalloc(&block_sum, sizeof(float)*grids);
  CalcEverything<<<grids, threads, sizeof(float) * (NUM_THREADS * 2 + 1)>>>(start, step, cuts, step/2, block_sum,dgpusum);
  
  
  
  #ifdef USE_CPU_REDUCTION
  float* rtns;
  rtns = new float[grids];
  cudaMemcpy(rtns, block_sum, sizeof(float)*grids, cudaMemcpyDeviceToHost);
  #pragma omp parallel for reduction(+:sum)
  for(int i=0;i<grids;i++){
    sum += rtns[i];
  }
  delete[]rtns;
  #endif

  #if defined USE_ATOMIC_SUM
  cudaMemcpy(gpu_sum, dgpusum, sizeof(float), cudaMemcpyDeviceToHost);
  sum = *gpu_sum;
  #endif

  #ifdef USE_GPU_SHARED_REDUCTION
  int cnt = grids;
  int gridcount;
  while(cnt!=1){
      gridcount = (cnt + NUM_THREADS - 1) / NUM_THREADS;
      SecondaryReduction<<<gridcount, NUM_THREADS,NUM_THREADS*sizeof(float)>>>(cnt, block_sum);
      cnt = gridcount;
  }
  cudaMemcpy(&sum, block_sum, sizeof(float), cudaMemcpyDeviceToHost);
  #endif

  cudaFree(dgpusum);
  cudaFree(block_sum);
  return sum;

}
float Target_Function2(float x){
  return TFUNCTION(x);
}
float calcByCPU(float start, float end, int cuts){
    double step = (end-start)/cuts;

    double sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < cuts; i++) {
        double x = start + i * step;
        sum += (Target_Function2(x) + Target_Function2(x + step)) * step / 2;
    }
    return (float)sum;
}

int main(int argc, char* argv[]) {
    float start = 0.0f;
    float end = 100.0f;
    int cuts = 819200000;
             //2147483647

    // CPU 시간 측정
    auto cpu_start = chrono::high_resolution_clock::now();
    float cpu_result = calcByCPU(start, end, cuts);
    auto cpu_end = chrono::high_resolution_clock::now();
    auto cpu_duration = chrono::duration_cast<chrono::microseconds>(cpu_end - cpu_start).count() / 1000.0; // 밀리초

    // GPU 시간 측정
    auto gpu_start = chrono::high_resolution_clock::now();
    float gpu_result = calcByGPU(start, end, cuts);
    cudaDeviceSynchronize();
    auto gpu_end = chrono::high_resolution_clock::now();
    auto gpu_duration = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count() / 1000.0; // 밀리초

    // 결과 출력
    cout << "CPU Result: " << cpu_result << ", Time: " << cpu_duration << " ms" << endl;
    cout << "GPU Result: " << gpu_result << ", Time: " << gpu_duration << " ms" << endl;

    return 0;
}

