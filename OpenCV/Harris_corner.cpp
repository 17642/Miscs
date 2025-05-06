#include<iostream>
#include"opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main() {
	string files[3] = {"chessboard.png","lena.bmp","fiigure.bmp"};

	Mat src = imread(files[2],IMREAD_GRAYSCALE);
	//Mat src = srcb.clone();
	//src.convertTo(src, CV_8UC1);
	Mat dst;

	int BlockSize = 8;
	int KernalSize = 1;
	double k = 0.06;      
	
	cornerHarris(src, dst, BlockSize, KernalSize, k, BORDER_DEFAULT);

	Mat dst2;
	normalize(dst, dst2,255);
	dst2.convertTo(dst2, CV_8UC1,255);
	//normalize 후 CV_8UC1 영상으로 변환한다.


	Mat result = src.clone();
	cvtColor(result, result, COLOR_GRAY2BGR);
	//영상에 색을 입혀야 하므로 포맷을 변환한다.

	for (int y = 0; y < result.rows; y++) {
		for (int x = 0; x < result.cols; x++) {
			if (dst2.at<uchar>(y, x) > 200) {
				//확실한 코너를 검출하기 위해 밝기값이 특히 높은 곳을 검출한다.
				result.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
			}
		}
	}

	

	imshow("dst2", dst2);
	imshow("result", result);

	waitKey(0);
	return 0;
}
