#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<stdlib.h>
#include<string.h>
#include<chrono>
#include<iostream>
  
#define NUM_DATA_MUL 1
#define NUM_DATA 1024*1024

__global__ void vecAdd(int* _a, int* _b, int* _c, int size){
  long long tID = blockIdx.y * ( gridDim.x * blockDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
  if(tID<NUM_DATA*size) _c[tID] = _a[tID] + _b[tID];
}

void calcByCPU(int* a, int* b, int* c, int size){
  for(int i=0; i<NUM_DATA*size; i++){
    c[i] = a[i] + b[i];
  }
}

void calcByGPU(int* a, int* b, int* c, int size){
  dim3 block_size(1024,1,1);
  dim3 grid_size(1024,size,1);
  vecAdd<<<grid_size,block_size>>>(a,b,c,size);
  cudaDeviceSynchronize();
}

int main(int argc, char* argv[]){

  int grid_size = argc==2?atoi(argv[1]):NUM_DATA_MUL;

  std::cout<<"data size: "<<NUM_DATA*grid_size<<std::endl;

  int *a, *b, *c, *p_c;
  int *d_a, *d_b, *d_c;

  std::chrono::duration<double, std::milli> cputime,gputime,HDtransfertime,DHtransfertime;

  a = new int[NUM_DATA*grid_size];
  b = new int[NUM_DATA*grid_size];
  c = new int[NUM_DATA*grid_size];
  p_c = new int[NUM_DATA*grid_size];

  for(int i=0; i<NUM_DATA*grid_size; i++){
    a[i] = rand()%10;
    b[i] = rand()%10;
  }

  auto startTime = std::chrono::high_resolution_clock::now();
  calcByCPU(a,b,c,grid_size);
  auto endTime = std::chrono::high_resolution_clock::now();
  cputime = endTime - startTime;

  cudaMalloc(&d_a, sizeof(int)*NUM_DATA*grid_size);
  cudaMalloc(&d_b, sizeof(int)*NUM_DATA*grid_size);
  cudaMalloc(&d_c, sizeof(int)*NUM_DATA*grid_size);

  startTime = std::chrono::high_resolution_clock::now();
  cudaMemcpy(d_a, a, sizeof(int)*NUM_DATA*grid_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(int)*NUM_DATA*grid_size, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_c, c, sizeof(int)*NUM_DATA, cudaMemcpyHostToDevice);
  endTime = std::chrono::high_resolution_clock::now();
  HDtransfertime = endTime - startTime;

  startTime = std::chrono::high_resolution_clock::now();
  calcByGPU(d_a, d_b, d_c,grid_size);
  endTime = std::chrono::high_resolution_clock::now();
  gputime = endTime - startTime;

  startTime = std::chrono::high_resolution_clock::now();
  cudaMemcpy(p_c, d_c, sizeof(int)*NUM_DATA*grid_size, cudaMemcpyDeviceToHost);
  endTime = std::chrono::high_resolution_clock::now();
  DHtransfertime = endTime - startTime;

  bool good = true;
  for(long long i = 0;i<NUM_DATA*grid_size;i++){
    if(p_c[i] != c[i]){
      good = false;
      std::cout<<"ERROR at "<<i<<std::endl;
      break;
    }
  }
  if(good){
    std::cout<<"CORRECT"<<std::endl;
  }

  std::cout<<"CPU TIME : "<<cputime.count()<<"ms"<<std::endl;
  std::cout<<"GPU TIME : "<<gputime.count()<<"ms"<<std::endl;
  std::cout<<"Host - Device TRANSFER TIME : "<<HDtransfertime.count()<<"ms"<<std::endl;
  std::cout<<"Device - Host TRANSFER TIME : "<<DHtransfertime.count()<<"ms"<<std::endl;

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  delete[] a;
  delete[] b;
  delete[] c;
  delete[] p_c;

  return 0;
}

