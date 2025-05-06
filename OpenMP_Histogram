#include<omp.h>
#include<iostream>
#include<random>
#include<stdio.h>
#include<stdlib.h>
#include"DS_timer.h"

#ifdef _WIN32
#include <Windows.h>
#else
#define Sleep(x) sleep(x)
#endif

#define INPUT_NUM 1024*1024 // 입력 수
#define NUM_THREADS 8 
#define RMIN 0 //입력 최소치
#define RMAX 10 // 입력 최대치
#define BIN_RANGE 1.0 // 분할 크기

#define DIVISION (int((RMAX - RMIN) / BIN_RANGE)) // 이때 각 분할의 수

using namespace std;

void print_result(int* rst, int len, string text) {
	cout << text << endl;
	for (long long i = 0; i < len-1; i++) {
		cout << rst[i] << "\t";
	}
	cout << endl;
}

float* make_float_list() {
	float* dat = new float[INPUT_NUM];

	random_device rd; //C++의 Random Device를 이용해 float 형의 균일한 난수를 생성한다.
	mt19937 gen(rd());
	uniform_real_distribution<float> rnd(RMIN, RMAX);

	for (long long i = 0; i < INPUT_NUM; i++) {
		dat[i] = rnd(gen);
	}

	return dat;
}

int* mk_histogram_serial(float* data) { // 직렬 알고리즘
	int* hist = new int[int((RMAX - RMIN) / BIN_RANGE) + 1];
	for (int i = 0; i<int((RMAX - RMIN) / BIN_RANGE) + 1; i++) { // 결과를 저장할 배열을 생성하고 초기화한다.
		hist[i] = 0;
	}

	for (long long i = 0; i < INPUT_NUM; i++) {
		hist[(int)((data[i] - RMIN) / BIN_RANGE)]++;
	}

	return hist; // 결과 배열을 리턴한다.
}

int* mk_histogram_v1(float* data) {
	int* hist = new int[int((RMAX - RMIN) / BIN_RANGE)];
	for (int i = 0; i<int((RMAX - RMIN) / BIN_RANGE); i++) {
		hist[i] = 0;
	}

#pragma omp parallel num_threads(NUM_THREADS) shared(hist) // 저장할 결과 배열을 공유로 지정한다.
	{
#pragma omp for 
		for (long long i = 0; i < INPUT_NUM; i++) {
#pragma omp atomic // atomic을 통해 값 증가에 문제가 생기지 않게 한다.
			hist[(int)((data[i] - RMIN) / BIN_RANGE)]++; // 이때 더하는 인덱스를 구한다.
		}

	}
	

	return hist;
}

int* mk_histogram_v2(float* data) {
	int* hist = new int[int((RMAX - RMIN) / BIN_RANGE)];
	int len = int((RMAX - RMIN) / BIN_RANGE) ; // 배열 길이
	for (int i = 0; i<len; i++) {
		hist[i] = 0;
	}

#pragma omp parallel num_threads(NUM_THREADS) shared(hist)
	{
		int* lbin = new int[len];
		for (int i = 0; i< len; i++) {
			lbin[i] = 0; //로컬 저장소를 만들고 초기화한다.
		}
#pragma omp for
		for (long long i = 0; i < INPUT_NUM; i++) {
			lbin[(int)((data[i] - RMIN) / BIN_RANGE)]++; 
		} // Ver 1과의 차이점은 전역에 직접 추가하는게 아닌 로컬 영역에 값을 증가하는 것이다.

		
#pragma omp critical // 값을 추가하는 부분은 critical을 통해 잘못 접근하지 않도록 한다
			for (int i = 0; i < len; i++) {
				hist[i] += lbin[i];
			}

			delete[]lbin; // 로컬 저장소는 할당 해제한다.
	}

	return hist;
}

int* mk_histogram_v3(float* data) {
	int* hist = new int[int((RMAX - RMIN) / BIN_RANGE)]; // 저장할 결과
	int** lbins = new int* [NUM_THREADS]; // 로컬 저장소이지만, 추후 옆 스레드의 데이터를 저장해야 하므로, 외부에 저장소를 만든다.
	int len = int((RMAX - RMIN) / BIN_RANGE); // 배열 길이

#pragma omp parallel num_threads(NUM_THREADS) shared(hist, lbins)
	{
		int tid = omp_get_thread_num(); // tid를 구하고 사용할 로컬 저장소를 지정한다
		lbins[tid] = new int[len];
		for (int i = 0; i<len; i++) {
			lbins[tid][i] = 0; // 로컬 저장소 생성 및 초기화
		}
#pragma omp for
		for (long long i = 0; i < INPUT_NUM; i++) {
			lbins[tid][(int)((data[i] - RMIN) / BIN_RANGE)]++;
		}

#pragma omp barrier//모든 연산이 끝날 때까지 대기한다.

		// tree reduction을 사용한다.

		for (int i = 1; i < NUM_THREADS; i *= 2) {  // i는 쓰레드 별로 2로 나누어가며 작동한다.
			for (int j = 0;j < len;j++) {
					if ((tid + j) % (i * 2) == 0) {
						lbins[tid][j] += lbins[(tid + i) % NUM_THREADS][j];
					}
			}

#pragma omp barrier // 각각 트리를 합칠 때 단계별로 모든 합성 작업이 완료될때까지 기다려야 한다.
		}

	}


	for (int i = 0;i < len;i++) {
		hist[i] = lbins[(NUM_THREADS - i%NUM_THREADS)%NUM_THREADS][i];
	} // 덧셈 과정에서 적절한 위치에 있는 값을 hist로 옮긴다.


	for (int i = 0;i < NUM_THREADS;i++) {
		delete[]lbins[i];
	}
	delete[]lbins; // 메모리 할당 해제한다.

	return hist; // 마지막으로 더해진 히스토그램만 삽입한다.
}

int main(int argc, int* argv[]) {

	int* hist_Serial, *hist_v1, *hist_v2, *hist_v3;
	float* datas = make_float_list();

	DS_timer timer(4);

	timer.setTimerName(0, (char*)"Serial");
	timer.setTimerName(1, (char*)"Version 1");
	timer.setTimerName(2, (char*)"Version 2");
	timer.setTimerName(3, (char*)"Version 3"); // 타이머 배정 및 세팅

	timer.onTimer(0);
	hist_Serial = mk_histogram_serial(datas);
	timer.offTimer(0);
	timer.onTimer(1);
	hist_v1 = mk_histogram_v1(datas);
	timer.offTimer(1);
	timer.onTimer(2);
	hist_v2 = mk_histogram_v2(datas);
	timer.offTimer(2);
	timer.onTimer(3);
	hist_v3 = mk_histogram_v3(datas);
	timer.offTimer(3);

	cout << "---Result---" << endl; // 실행 후 결과 출력

	int hist_range = (int)((RMAX - RMIN) / BIN_RANGE) + 1;
	for (int i = 0;i < hist_range - 1;i++) {
		cout << RMIN + BIN_RANGE * i << "~" << RMIN + BIN_RANGE * (i + 1) << "\t";
	} // 미리 결과에서 히스토그램이 나타내는 범위를 출력한다.
	cout << endl;
	print_result(hist_Serial, hist_range, "--Serial--");
	print_result(hist_v1, hist_range, "--Version 1--");
	print_result(hist_v2, hist_range, "--Version 2--");
	print_result(hist_v3, hist_range, "--Version 3--");
	cout << endl;
	timer.printTimer();

	return 0;
}

