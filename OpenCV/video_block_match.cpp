#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
  int findRectSize = 30;//ROI는 Template의 rows,cols에 각각 findRectSIze를 더한 크기를 갖는다.
	Mat ref = imread("crs/Template_Car_100.png");
	int firstFrame = 100;
	int frm = firstFrame; // 파일이 00100부터 시작하므로 첫 프레임을 100으로 지정한다.
	string targetFile = "crs/00" + to_string(frm) + ".jpg"; 
	Mat Frame = imread(targetFile);
	bool isFirst = true;
	Rect poi(0,0,0,0);//초기에는 roi를 찾을 수 없으므로 0으로 지정한다.
	while (!Frame.empty()) {

		Mat dst = Mat(Size(Frame.cols - ref.cols + 1, Frame.rows - ref.rows + 1), CV_32FC1,1);
		Mat dst2 = Mat(Size(findRectSize+1, findRectSize+1), CV_32FC1,1);
		//matchTemplate(Frame, ref, dst, TM_CCOEFF_NORMED);
		//ROI가 있는 경우와 없는 경우의 탐색 범위는 차이가 있으므로 배열을 다르게 지정


		if (isFirst) {
			matchTemplate(Frame, ref, dst, TM_CCOEFF_NORMED);
		}
		else {
			Mat poim = Mat(Size(findRectSize + ref.cols, findRectSize + ref.rows), ref.type());
			poim = Frame(poi);
			matchTemplate(poim, ref, dst2, TM_CCOEFF_NORMED);
		}


		double maxVal;
		Point maxPoint;
		minMaxLoc(dst, 0, &maxVal, 0, &maxPoint);
		if (!isFirst) minMaxLoc(dst2, 0, &maxVal, 0, &maxPoint);
		if(!isFirst)maxPoint.x += poi.x;
		if(!isFirst)maxPoint.y += poi.y;//첫 프레임에서는 Roi의 위치를 찾아야 하므로 ROI 관련 연산을 하지 않는다.
		poi = Rect(maxPoint.x - findRectSize / 2, maxPoint.y - findRectSize / 2, findRectSize + ref.cols, findRectSize + ref.rows);
		poi.x = max(1, min(Frame.cols - (findRectSize + ref.cols)-1, poi.x));
		poi.y = max(1, min(Frame.rows - (findRectSize + ref.rows)-1, poi.y));
		cout <<"Frame " << frm - firstFrame << ": " << maxPoint.x+ref.cols/2 << "," << maxPoint.y+ref.rows/2 << endl; //프레임별 찾은 영역 좌표
		//circle(Frame, Point(maxPoint.x + ref.cols / 2, maxPoint.y + ref.rows / 2), 2, Scalar(255, 255, 0),2); //찾은 영역의 중점 (Cyan)
		//rectangle(Frame, poi, Scalar(0, 255, 0)); // ROI 영역 (Green)
		rectangle(Frame, maxPoint, Point(maxPoint.x + ref.cols, maxPoint.y + ref.rows), Scalar(0, 0, 255)); //찾은 영역 (Red)
		
		imshow("frame", Frame);
		waitKey(16); // 초당 60프레임
		frm++;
		targetFile = "crs/00" + to_string(frm) + ".jpg"; // 다음 파일 이름 생성 
		Frame = imread(targetFile);

		if (isFirst)isFirst = false;
	}
	waitKey(0);
	cout << "END" << endl;


	return 0;
}
