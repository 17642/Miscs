#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "DS_timer.h"
#include "DS_definitions.h"

#define NUM_THREADS 24

int main(int argc, char* argv[]) {

	DS_timer timer(2);
	timer.setTimerName(0, (char*)"Serial");
	timer.setTimerName(1, (char*)"Parallel");
	//타이머 설정

	double min = atof(argv[1]);
	double max = atof(argv[2]);

	long long n = atoi(argv[3]);
	double div = (max - min) / (double)n;

	double serialResult=0;
	double parallelResult=0;
	//매개변수 및 변수 설정

	const char* target = "x*x";

	

	timer.onTimer(0);
	for (long long i = 0; i < n; i++) {
	
		serialResult += ((min + i * div) * (min + i * div) + (min + (i + 1) * div) * (min + (i + 1) * div)) * div / 2.0;
	}
	//직렬 연산
	timer.offTimer(0);

	timer.onTimer(1);
	double* fig = new double[n];
	//값을 모으기 위한 배열

#pragma omp parallel num_threads(NUM_THREADS)
	{
#pragma omp for
		for (long long i = 0; i < n; i++) {

			fig[i] += ((min + i * div) * (min + i * div) + (min + (i + 1) * div) * (min + (i + 1) * div)) * div / 2.0;
		}
		//병렬 연산
	}

	for (int i = 0; i < n; i++)
	{

		parallelResult += fig[i];
	}
	//값은 모든 연산을 끝낸 후 취합한다.


	delete[]fig;
	timer.offTimer(1);




	//출력
	printf("f(x) = x * x\n");
	printf("range = (%lf, %lf), n = %ld\n",min,max,n);
	printf("[Serial] area = %lf\n", serialResult);
	printf("[Parallel] area = %lf\n", parallelResult);
	timer.printTimer();

	return 0;
}
