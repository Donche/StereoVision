// CameraCalibration.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "StereoVision.h"
#include "StereoCalibration.h"
#include "TakePictures.h"

#define IMAGE_HEIGHT 777
#define IMAGE_WIDTH 1000
#define DISP_IMG_HEIGHT 400
#define DISP_IMG_WIDTH 500

using namespace std;
using namespace cv;

void openCamera() {
	VideoCapture cam1(1);
	VideoCapture cam0(0);
	if (!cam1.isOpened() || !cam0.isOpened()) {
		cout << "camera not opened!" << endl;
		getchar();
		return;
	}
	cam0.set(CV_CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH);
	cam0.set(CV_CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT);
	cam1.set(CV_CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH);
	cam1.set(CV_CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT);
	Mat imgLeft, imgRight, imgTotal;
	Rect rect1 = Rect(0, 0, DISP_IMG_WIDTH, DISP_IMG_HEIGHT);
	Rect rect2 = Rect(DISP_IMG_WIDTH, 0, DISP_IMG_WIDTH, DISP_IMG_HEIGHT);
	while (1) {
		cam0 >> imgLeft;
		cam1 >> imgRight;
		imgTotal.create(Size(DISP_IMG_WIDTH * 2, DISP_IMG_HEIGHT), CV_8UC3);
		resize(imgLeft, imgTotal(rect1), Size(DISP_IMG_WIDTH, DISP_IMG_HEIGHT));
		resize(imgRight, imgTotal(rect2), Size(DISP_IMG_WIDTH, DISP_IMG_HEIGHT));
		imshow("img", imgTotal);
		waitKey(30);
	}
}

int main()
{
	char inp, inp2;

	TakePictures tp;
	StereoCalibration sc;
	StereoVision sv;
	while (1)
	{
		cout << "1. Take Photos \t 2.Stereo Calibrate \t 3.Run Stereo Vision \t "
			<<"4.Just Open the Camera \t 0.Exit" << endl;
		cin >> inp;

		switch (inp) {
		case '1':
			tp.runTakePictures();
			break;
		case '2':
			sc.runStereoCalibration();
			break;
		case '3':
			if(sv.initSV())	sv.runVision();
			break;
		case '4':
			openCamera();
			break;
		case '\n':
			break;
		case '0':
			return 0;
		default:
			cout << "wwwwwwwwwhat??" << endl;
			break;
		}
	}
}

