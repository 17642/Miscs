#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

int main() {
	Mat src = imread("00001.jpg",IMREAD_GRAYSCALE);
	//흑백으로 이미지를 읽어야 최적으로 작동한다.
	Mat defX, defY;

	auto imgSize = src.size;

	Sobel(src, defX, CV_32F, 0, 1);
	Sobel(src, defY, CV_32F, 1, 0);
	//X축과 Y축에 대해 미분을 진행한다.
	//cartToPolar 함수는 32F에서 잘 작동한다.

	Mat FilteredEdge = Mat::zeros(src.size(), CV_8U);
	Mat Magnitude, Angle;
	cartToPolar(defX, defY, Magnitude, Angle, true);
	//각도를 도 단위로출력한다
	//cartToPolar 함수는 각 미분을 기반으로 크기와 방향을 계산한다.

	float leftMin = 30;
	float leftMax = 50;
	float rightMin = 125;
	float rightMax = 150;

	float magThresHold = 20.0;
	//추출을 원하는 각도의 범위를 지정한다.

	for (int i = 0;i < Angle.rows;i++) {
		for (int j = 0;j < Angle.cols;j++) {
			float currentAngle = Angle.at<float>(i, j);
			float mag = Magnitude.at<float>(i,j);
			//Angle과 Magnitude 배열에서 값을 추출한다.

			if (j<Angle.cols / 2 && (currentAngle >= leftMin && currentAngle <= leftMax) && mag>magThresHold) {
				FilteredEdge.at<uchar>(i, j) = saturate_cast<int>(mag); // 좌측 추출
			}
			else if((j>Angle.cols/2)&&(currentAngle >= rightMin && currentAngle <= rightMax) && mag > magThresHold) {
				FilteredEdge.at<uchar>(i, j) = saturate_cast<int>(mag); // 우측 추출
			}


		}
	}

	//HoughLinesP로 직선 검출 (시작점과 끝점 포함)
	vector<Vec4i> lines;
	HoughLinesP(FilteredEdge, lines, 1, CV_PI / 180, 80, 30, 100);

	//직선 길이 계산
	vector<pair<double, Vec4i>> lineLengths; // 직선 길이 리스트
	for (const auto& line : lines) {
		double length = sqrt(pow(line[2] - line[0], 2) + pow(line[3] - line[1], 2));
		lineLengths.push_back({ length, line }); //라인 길이를 구해서 길이 리스트에 삽입한다.
	}

	sort(lineLengths.begin(), lineLengths.end(), [](const auto& a, const auto& b) {
		return a.first > b.first;
		}); // 정렬

	Mat laneEdge = Mat::zeros(src.size(), CV_8U);
	int linesToDraw = min(6, static_cast<int>(lineLengths.size())); // 최대 6개
	for (int i = 0; i < linesToDraw; i++) {
		Vec4i l = lineLengths[i].second;
		line(laneEdge, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 1, LINE_AA);//빈 이미지에 선을 그려주자.
	}


	Mat dst = FilteredEdge & laneEdge; // 출력은 추출된 직선과 검출된 라인을 and 연산한 결과.
	//equalizeHist(dst, dst); // 평활화를 수행한다.
	threshold(dst, dst, 100, 255, THRESH_OTSU);//아니면 이진화를 수행하거나

	
	

	imshow("result",dst);
	waitKey(0);


	return 0;
}
