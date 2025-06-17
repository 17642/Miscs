#include <iostream>
#include <map>
#include "opencv2/opencv.hpp"

#define USE_CONTROL_BAR
#define USE_FIXED_BOX_SIZE

using namespace std;
using namespace cv;

#define SPEED_MUL 1
#define MORPH1_SIZE 17
#define MORPH2_SIZE 5
#define HMAX 100
#define HMIN 70
#define SMIN 70
#define SMAX 255
#define VMIN 100
#define VMAX 255

#define FIXED_BOX_SIZE 25.0

#define VIDEO_PATH "Tennis_ball2.mp4"
#define SAVE_PATH "tracked_vid.mp4"

// ROI 크기 제한 상수
#define MIN_ROI_SIZE 2   // 최소 너비/높이
#define MAX_ROI_SIZE 15  // 최대 너비/높이

// 정적 ROI 판단 기준
#define STATIC_DIST_THRESHOLD 1.0  // 프레임 간 이동 거리 임계값 (픽셀)
#define STATIC_FRAME_THRESHOLD 2   // 정적 상태로 판단하는 최소 프레임 수
#define SHOW_FRAME_COUNT 0         // 물체가 n 프레임 이상 유지되었을 경우에만 표시
#define MAX_FRAMES_LOST 10         // ROI가 감지되지 않아도 유지되는 프레임 수
#define START_ROI_SIZE 5           // 컨투어 최소 면적
#define MOVEMENT_DIST_THRESHOLD 2.0 // 프레임 간 이동 거리 값이 한번이라도 이 값일 경우 공으로 판단

// ROI와 ID를 함께 저장하는 구조체
struct TrackedROI {
    Rect roi;
    int id;
    int framesSinceLastSeen; // 마지막으로 감지된 이후 지난 프레임 수
    Point2f lastCenter;      // 이전 프레임의 중심 좌표
    int staticFrameCount;    // 정적 상태로 유지된 프레임 수
    int usedFrameCount;      // 유지된 프레임 수
    bool isBall;             // 공 여부 플래그
};

TrackedROI* findROIbyID(int id, vector<TrackedROI>& lst) {
    for (auto &tracked : lst) {
        if (tracked.id == id) {
            return &tracked;
        }
    }
    return nullptr;
}

// 중심 거리 계산
double calcCenterDistance(const Rect& r1, const Rect& r2) {
    Point2f c1(r1.x + r1.width / 2.0f, r1.y + r1.height / 2.0f);
    Point2f c2(r2.x + r2.width / 2.0f, r2.y + r2.height / 2.0f);
    return norm(c1 - c2);
}

Point2f getCenterPoint(const Rect& r) {
    return Point2f(r.x + r.width / 2.0f, r.y + r.height / 2.0f);
}

// 중심점이 Bounding Box 내에 있는지 확인
bool isCenterWithinBoundingBox(const Rect& roi, const Rect& boundingBox) {
    Point2f center(roi.x + roi.width / 2.0f, roi.y + roi.height / 2.0f);
    return (center.x >= boundingBox.x &&
        center.y >= boundingBox.y &&
        center.x <= (boundingBox.x + boundingBox.width) &&
        center.y <= (boundingBox.y + boundingBox.height));
}

// 모든 후보 ROI를 탐색하고 이전 ROI와 매칭, 새로운 ROI 추가
vector<TrackedROI> findAndTrackROIs(const Mat& binaryMask, vector<TrackedROI>& prevROIs, const Rect& boundingBox, double maxDist = 60.0, int maxFramesLost = MAX_FRAMES_LOST) {
    vector<vector<Point>> contours;
    findContours(binaryMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Rect> candidates;
    for (const auto& c : contours) {
        double area = contourArea(c);
        if (area < START_ROI_SIZE) continue; // 너무 작은 영역 제외
        Rect r = boundingRect(c);
        // ROI 크기 제한
        if (r.width < MIN_ROI_SIZE || r.height < MIN_ROI_SIZE || r.width > MAX_ROI_SIZE || r.height > MAX_ROI_SIZE) continue;
        // 중심점이 Bounding Box 내에 있는지 확인
        if (!isCenterWithinBoundingBox(r, boundingBox)) continue;
        candidates.push_back(r);
    }

    vector<TrackedROI> matchedROIs;
    vector<bool> usedCandidates(candidates.size(), false);
    static int nextID = 0; // 고유 ID 생성용

    // 이전 ROI와 후보군 매칭
    for (auto& prevROI : prevROIs) {
        double bestDist = maxDist;
        int bestCandIdx = -1;

        for (size_t i = 0; i < candidates.size(); ++i) {
            if (usedCandidates[i]) continue;
            double dist = calcCenterDistance(prevROI.roi, candidates[i]);
            if (dist < bestDist) {
                bestDist = dist;
                bestCandIdx = i;
            }
        }

        if (bestCandIdx != -1) {
            // 매칭된 경우
            Point2f newCenter(candidates[bestCandIdx].x + candidates[bestCandIdx].width / 2.0f,
                candidates[bestCandIdx].y + candidates[bestCandIdx].height / 2.0f);
            int newStaticCount = prevROI.staticFrameCount;

            int newUsedCount = prevROI.usedFrameCount + 1; // 유지된 프레임 수 증가

            if (newStaticCount > STATIC_FRAME_THRESHOLD) {
                newUsedCount = 0;
            }
            bool isBall = prevROI.isBall;
            // 이동 거리 확인
            double dist = norm(newCenter - prevROI.lastCenter);
            if (dist >= MOVEMENT_DIST_THRESHOLD) {
                isBall = true; // 이동 거리가 임계값 이상이면 공으로 판단
            }
            if (dist < STATIC_DIST_THRESHOLD) {
                newStaticCount++;
            }
            else {
                newStaticCount = 0; // 움직였으므로 정적 카운트 리셋
            }
            matchedROIs.push_back({ candidates[bestCandIdx], prevROI.id, 0, newCenter, newStaticCount, newUsedCount, isBall });
            usedCandidates[bestCandIdx] = true;
        }
        else {
            // 매칭되지 않은 경우 framesSinceLastSeen 증가 및 usedFrameCount 리셋
            prevROI.framesSinceLastSeen++;
            //prevROI.usedFrameCount = 0; // ROI를 놓치면 usedFrameCount 리셋
            
            
            if (prevROI.framesSinceLastSeen <= maxFramesLost) {
                matchedROIs.push_back(prevROI); // 일정 프레임 동안 유지
            }
        }
    }

    // 매칭되지 않은 새로운 후보군을 새로운 ROI로 추가
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (!usedCandidates[i]) {
            Point2f center(candidates[i].x + candidates[i].width / 2.0f, candidates[i].y + candidates[i].height / 2.0f);
            matchedROIs.push_back({ candidates[i], nextID++, 0, center, 0, 1, false }); // 새로운 ROI는 공이 아닌 상태로 시작
        }
    }

    return matchedROIs;
}

// 첫 프레임에서 모든 후보 ROI 찾기
vector<TrackedROI> findFirstROIs(const Mat& binaryMask, const Rect& boundingBox) {
    vector<vector<Point>> contours;
    findContours(binaryMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<TrackedROI> rois;
    static int nextID = 0;

    for (const auto& c : contours) {
        double area = contourArea(c);
        if (area < START_ROI_SIZE) continue; // 너무 작은 영역 제외
        Rect r = boundingRect(c);
        // ROI 크기 제한
        if (r.width < MIN_ROI_SIZE || r.height < MIN_ROI_SIZE || r.width > MAX_ROI_SIZE || r.height > MAX_ROI_SIZE) continue;
        // 중심점이 Bounding Box 내에 있는지 확인
        if (!isCenterWithinBoundingBox(r, boundingBox)) continue;
        Point2f center(r.x + r.width / 2.0f, r.y + r.height / 2.0f);
        rois.push_back({ r, nextID++, 0, center, 0, 1, false }); // 새로운 ROI는 공이 아닌 상태로 시작
    }

    return rois;
}

int main() {
    VideoCapture vid(VIDEO_PATH);
    if (!vid.isOpened()) {
        cout << "Error opening video file" << endl;
        return -1;
    }


    VideoWriter vw;
    
    Mat frame, hsv;
    int frame_counter = 0;

    //영상 정보 획득
    int height = vid.get(CAP_PROP_FRAME_HEIGHT);
    int width = vid.get(CAP_PROP_FRAME_WIDTH);
    double fps = vid.get(CAP_PROP_FPS);
    int frame_count = vid.get(CAP_PROP_FRAME_COUNT);

    vw.open(SAVE_PATH, VideoWriter::fourcc('H', '2', '6', '4'), fps, Size(width, height), true);
    if (!vw.isOpened())
    {
        std::cout << "Can't write video !!! check setting" << std::endl;
        return -1;
    }

    // 정적 Bounding Box 설정 (Left Top: (360, 200), Right Bottom: (width-700, height-400))
    Rect boundingBox(360, 200, width - 700, height - 400);

    //수동으로 조작할 경우 조작용 창 생성
#ifdef USE_CONTROL_BAR
    namedWindow("Control", WINDOW_AUTOSIZE);
    int hmin = HMIN, smin = SMIN, vmin = VMIN, hmax = HMAX, smax = SMAX, vmax = VMAX;
    int bb_x = 360, bb_y = 200, bb_width = width - 700, bb_height = height - 400;

    createTrackbar("H_min", "Control", &hmin, 179);
    createTrackbar("H_max", "Control", &hmax, 179);
    createTrackbar("S_min", "Control", &smin, 255);
    createTrackbar("S_max", "Control", &smax, 255);
    createTrackbar("V_min", "Control", &vmin, 255);
    createTrackbar("V_max", "Control", &vmax, 255);
    createTrackbar("BB_x", "Control", &bb_x, width);
    createTrackbar("BB_y", "Control", &bb_y, height);
    createTrackbar("BB_width", "Control", &bb_width, width);
    createTrackbar("BB_height", "Control", &bb_height, height);
#endif

    vector<TrackedROI> currentROIs;
    bool foundFirstROI = false;

    while (true) {
        vid >> frame;
        if (frame.empty()) break;

        cvtColor(frame, hsv, COLOR_BGR2HSV);
        Mat rst, rst2;
        //GaussianBlur(hsv, hsv, Size(5, 5), 0);
#ifdef USE_CONTROL_BAR
        inRange(hsv, Scalar(hmin, smin, vmin), Scalar(hmax, smax, vmax), rst);
        // 트랙바로 Bounding Box 업데이트
        boundingBox = Rect(bb_x, bb_y, bb_width, bb_height);
#else
        inRange(hsv, Scalar(HMIN, SMIN, VMIN), Scalar(HMAX, SMAX, VMAX), rst);
#endif
        //GaussianBlur(rst, rst, Size(5, 5), 0);
        //threshold(rst, rst, 25, 255, THRESH_BINARY);
        //공 분류를 위한 모폴로지 연산
        Mat kernal = getStructuringElement(MORPH_ELLIPSE, Size(MORPH1_SIZE, MORPH1_SIZE));
        Mat kernal2 = getStructuringElement(MORPH_ELLIPSE, Size(MORPH2_SIZE, MORPH2_SIZE));
        morphologyEx(rst, rst2, MORPH_CLOSE, kernal, Point(-1, -1), 1);
        morphologyEx(rst2, rst2, MORPH_OPEN, kernal2);

        if (!foundFirstROI) {
            currentROIs = findFirstROIs(rst2, boundingBox);
            if (!currentROIs.empty()) {
                foundFirstROI = true;
                cout << "First ROIs found: " << currentROIs.size() << endl;
            }
        }
        else {
            currentROIs = findAndTrackROIs(rst2, currentROIs, boundingBox, 60.0, MAX_FRAMES_LOST);
            if (currentROIs.empty()) {
                cout << "Lost track of all ROIs" << endl;
                foundFirstROI = false;
            }
        }

        // Bounding Box 표시
        //rectangle(frame, boundingBox, Scalar(255, 0, 0), 1); // 파란색으로 표시

        int maxUFCID = -1;

        // 움직이고 SHOW_FRAME_COUNT 이상 유지된 공으로 판단된 ROI만 표시
        for (const auto& trackedROI : currentROIs) {
            if (trackedROI.staticFrameCount < STATIC_FRAME_THRESHOLD &&
                trackedROI.usedFrameCount >= SHOW_FRAME_COUNT &&
                trackedROI.isBall) {
                maxUFCID = (maxUFCID == -1) ? trackedROI.id : ((findROIbyID(maxUFCID, currentROIs) != nullptr && findROIbyID(maxUFCID, currentROIs)->usedFrameCount > trackedROI.usedFrameCount) ? maxUFCID : trackedROI.id);
                //rectangle(frame, trackedROI.roi, Scalar(0, 255, 0), 2);
                // ID, 정적 프레임 수, 유지된 프레임 수, 공 여부 표시
                string idText = "ID: " + to_string(trackedROI.id) + ", S: " + to_string(trackedROI.staticFrameCount) +
                    ", U: " + to_string(trackedROI.usedFrameCount) + ", Ball: " + (trackedROI.isBall ? "Y" : "N");
               // putText(frame, idText, Point(trackedROI.roi.x, trackedROI.roi.y - 10),FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
            }
        }

        TrackedROI* roiptr = findROIbyID(maxUFCID, currentROIs);
        if (roiptr != nullptr) {
#ifndef USE_FIXED_BOX_SIZE
            rectangle(frame, findROIbyID(maxUFCID, currentROIs)->roi, Scalar(0, 0, 255), 2);
#else
            Point2f ct = getCenterPoint(findROIbyID(maxUFCID, currentROIs)->roi);
            Point tl((int)(ct.x - FIXED_BOX_SIZE/2), (int)(ct.y -FIXED_BOX_SIZE/2));
            Rect newRect(tl, Size(FIXED_BOX_SIZE, FIXED_BOX_SIZE));
            rectangle(frame, newRect, Scalar(0, 0, 255), 2);
#endif
        }
        

        cout << "FRAME: " << frame_counter++ <<endl;
        cout << "TRACKING ROI INFO: " << "ID: " << maxUFCID << endl;
        if (roiptr == nullptr) {
            cout << "NONE" << endl;
        }
        else {
            cout << "ROI: " << roiptr->roi << ", U: " << roiptr->usedFrameCount << endl;
        }

#ifdef USE_CONTROL_BAR
        imshow("vid", rst2);
#endif
        imshow("frame_with_roi", frame);

        vw << frame;

        if (waitKey(1000 * SPEED_MUL / fps) == 27) break;
    }

    cout << "CAPTURE END" << endl;
    return 0;
}
