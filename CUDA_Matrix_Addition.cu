#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<stdlib.h>
#include<string.h>
#include<chrono>
#include<iostream>
#include<cmath>

using namespace std;

#define MatX 8192
#define MatY 8192

#define DefaultBlockSize 1024

//1D to 2D
#define _row(x,N) (x/N)
#define _col(y,N) (y%N)

#define _ID(x, y, N) ((x) * (N) + (y))

void calcByCPU(int* a, int* b, int* c){
  for(int y=0; y<MatY; y++){
    for(int x=0; x<MatX; x++){
      c[_ID(x,y,MatX)] = a[_ID(x,y,MatX)] + b[_ID(x,y,MatX)];
    }
  }
}

pair<int,int>splicBlockSize(int block_size){
  if(block_size==1){
    return pair<int,int>(1,1);
  }
  int log2N = log2(block_size);
  int k = log2N/2;
  return pair<int,int>(pow(2,k),pow(2,log2N%2==0?k:k+1));
}

__global__ void vecAdd1(int* _a, int* _b, int* _c, int size){
  int get_thread_id =  threadIdx.y*blockDim.x+threadIdx.x;
  long long tID = blockIdx.y * ( gridDim.x * blockDim.x * blockDim.y) + blockIdx.x * blockDim.x*blockDim.y + get_thread_id;
  if(tID<MatX*MatY) _c[tID] = _a[tID] + _b[tID];
}

__global__ void vecAdd3(int* _a, int* _b, int* _c, int size){
  long long tID = blockIdx.y * ( gridDim.x * blockDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
  int row = _row(tID,MatX);
  int col = _col(tID,MatX);
  if(row<MatY && col<MatX) _c[_ID(col,row,MatX)] = _a[_ID(col,row,MatX)] + _b[_ID(col,row,MatX)];
}

__global__ void vecAdd2(int* _a, int* _b, int* _c, int size){
  long long tID =  blockIdx.x * blockDim.x + threadIdx.x;
  if(tID<MatX*MatY) _c[tID] = _a[tID] + _b[tID];
}

void calcByGPU1(int* a, int* b, int* c, int block_size){ //2D GRID with 2D Blocks
  pair<int,int> gs = splicBlockSize(block_size);
  dim3 GridSize(MatX/gs.first,MatY/gs.second,1);
  dim3 BlockSize(gs.first,gs.second,1);

  vecAdd1<<<GridSize,BlockSize>>>(a,b,c,block_size);
  cudaDeviceSynchronize();
}

void calcByGPU2(int* a, int* b, int* c, int block_size){ //1D GRID with 1D Blocks
  dim3 GridSize(MatX*MatY/block_size,1,1);
  dim3 BlockSize(block_size,1,1);
  vecAdd2<<<GridSize,BlockSize>>>(a,b,c,block_size);
  cudaDeviceSynchronize();
}

void calcByGPU3(int* a, int* b, int* c, int block_size){ // 2D GRID with 1D Blocks
  pair<int,int> gs = splicBlockSize(block_size);
  dim3 GridSize(MatX/gs.first,MatY/gs.second,1);
  dim3 BlockSize(block_size,1,1);

  vecAdd3<<<GridSize,BlockSize>>>(a,b,c,block_size);
  cudaDeviceSynchronize();
}

void copyToGpu(int* a, int* b, int* d_a, int* d_b){
  cudaMemcpy(d_a, a, sizeof(int)*MatX*MatY, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(int)*MatX*MatY, cudaMemcpyHostToDevice);
}

bool checkGood(int*a, int*b, int index){
  for(int y=0; y<MatY; y++){
    for(int x=0; x<MatX; x++){
      if(a[_ID(x,y,MatX)] != b[_ID(x,y,MatX)]){
        cout<<index<<" ERROR at "<<_ID(x,y,MatX)<<endl;
        return false;
      }
    }
  }
  return true;
}

int main(int argc, char* argv[]){

  int *a, *b, *c;
  int *d1_a, *d1_b, *d1_c;
  int *d2_a, *d2_b, *d2_c;
  int *d3_a, *d3_b, *d3_c;
  int *rt1, *rt2, *rt3;

  int block_size = argc==2?atoi(argv[1]):DefaultBlockSize;

  std::chrono::duration<double, std::milli> cputime;
  std::chrono::duration<double, std::milli> gputime[3];
  std::chrono::duration<double, std::milli> HDtransfertime[3];
  std::chrono::duration<double, std::milli> DHtransfertime[3];

  a = new int[MatX*MatY];
  b = new int[MatX*MatY];
  c = new int[MatX*MatY];

  rt1 = new int[MatX*MatY];
  rt2 = new int[MatX*MatY];
  rt3 = new int[MatX*MatY];

  cudaMalloc(&d1_a, sizeof(int)*MatX*MatY);
  cudaMalloc(&d1_b, sizeof(int)*MatX*MatY);
  cudaMalloc(&d1_c, sizeof(int)*MatX*MatY);

  cudaMalloc(&d2_a, sizeof(int)*MatX*MatY);
  cudaMalloc(&d2_b, sizeof(int)*MatX*MatY);
  cudaMalloc(&d2_c, sizeof(int)*MatX*MatY);

  cudaMalloc(&d3_a, sizeof(int)*MatX*MatY);
  cudaMalloc(&d3_b, sizeof(int)*MatX*MatY);
  cudaMalloc(&d3_c, sizeof(int)*MatX*MatY);

  cout<<"block_size: "<<block_size<<endl;

  for(int y=0; y<MatY; y++){
    for(int x=0; x<MatX; x++){
      a[_ID(x,y,MatX)] = rand()%10;
      b[_ID(x,y,MatX)] = rand()%10;
    }
  }
  cout<<"\nARRAY SET"<<endl;

  auto startTime = std::chrono::high_resolution_clock::now();
  calcByCPU(a,b,c);
  auto endTime = std::chrono::high_resolution_clock::now();
  cputime = endTime - startTime;

  cout<<"CPU END\n"<<endl;

  startTime = std::chrono::high_resolution_clock::now();
  copyToGpu(a,b,d1_a,d1_b);
  endTime = std::chrono::high_resolution_clock::now();
  HDtransfertime[0] = endTime - startTime;

  startTime = std::chrono::high_resolution_clock::now();
  calcByGPU1(d1_a, d1_b, d1_c, block_size);
  endTime = std::chrono::high_resolution_clock::now();
  gputime[0] = endTime - startTime;

  startTime = std::chrono::high_resolution_clock::now();
  cudaMemcpy(rt1, d1_c, sizeof(int)*MatX*MatY, cudaMemcpyDeviceToHost);
  endTime = std::chrono::high_resolution_clock::now();
  DHtransfertime[0] = endTime - startTime;

  startTime = std::chrono::high_resolution_clock::now();
  copyToGpu(a,b,d2_a,d2_b);
  endTime = std::chrono::high_resolution_clock::now();
  HDtransfertime[1] = endTime - startTime;

  startTime = std::chrono::high_resolution_clock::now();
  calcByGPU2(d2_a, d2_b, d2_c, block_size);
  endTime = std::chrono::high_resolution_clock::now();
  gputime[1] = endTime - startTime;

  startTime = std::chrono::high_resolution_clock::now();
  cudaMemcpy(rt2, d2_c, sizeof(int)*MatX*MatY, cudaMemcpyDeviceToHost);
  endTime = std::chrono::high_resolution_clock::now();
  DHtransfertime[1] = endTime - startTime;

  startTime = std::chrono::high_resolution_clock::now();
  copyToGpu(a,b,d3_a,d3_b);
  endTime = std::chrono::high_resolution_clock::now();
  HDtransfertime[2] = endTime - startTime;

  startTime = std::chrono::high_resolution_clock::now();
  calcByGPU3(d3_a, d3_b, d3_c, block_size);
  endTime = std::chrono::high_resolution_clock::now();
  gputime[2] = endTime - startTime;

  startTime = std::chrono::high_resolution_clock::now();
  cudaMemcpy(rt3, d3_c, sizeof(int)*MatX*MatY, cudaMemcpyDeviceToHost);
  endTime = std::chrono::high_resolution_clock::now();
  DHtransfertime[2] = endTime - startTime;

  if(checkGood(rt1,c,1)){
    cout<<"CORRECT 1"<<endl;
  }
  if(checkGood(rt2,c,2)){
    cout<<"CORRECT 2"<<endl;
  }
  if(checkGood(rt3,c,3)){
    cout<<"CORRECT 3"<<endl;
  }


  cout<<"\nCPU TIME : "<<cputime.count()<<"ms"<<endl;
  cout<<"GPU TIME 1: "<<gputime[0].count()<<"ms"<<endl;
  cout<<"GPU TIME 2: "<<gputime[1].count()<<"ms"<<endl;
  cout<<"GPU TIME 3: "<<gputime[2].count()<<"ms\n"<<endl;

  cout<<"Host - Device TRANSFER TIME 1: "<<HDtransfertime[0].count()<<"ms"<<endl;
  cout<<"Host - Device TRANSFER TIME 2: "<<HDtransfertime[1].count()<<"ms"<<endl;
  cout<<"Host - Device TRANSFER TIME 3: "<<HDtransfertime[2].count()<<"ms\n"<<endl;

  cout<<"Device - Host TRANSFER TIME 1: "<<DHtransfertime[0].count()<<"ms"<<endl;
  cout<<"Device - Host TRANSFER TIME 2: "<<DHtransfertime[1].count()<<"ms"<<endl;
  cout<<"Device - Host TRANSFER TIME 3: "<<DHtransfertime[2].count()<<"ms\n"<<endl;

  cudaFree(d1_a);
  cudaFree(d1_b);
  cudaFree(d1_c);

  cudaFree(d2_a);
  cudaFree(d2_b);
  cudaFree(d2_c);

  cudaFree(d3_a);
  cudaFree(d3_b);
  cudaFree(d3_c);

  delete[] a;
  delete[] b;
  delete[] c;

  delete[] rt1;
  delete[] rt2;
  delete[] rt3;

  return 0;
}
