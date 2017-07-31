#pragma once

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/legacy/legacy.hpp>


using cv::Mat;
using cv::Size;
using cv::Rect;

class StereoVision
{
public:
	StereoVision();
	~StereoVision();

	bool initSV();
	void runVision();
	bool runStereoVisionBM();
	bool runPhotoStereoVisionBM();
	bool runStereoVisionSGBM();
	bool runStereoPhotoSIFT();
	bool runStereoVisionSIFT();

	void BM23D(Mat& disparity32F);

private:
	Mat Camera_Matrix0, distCoeff0, Camera_Matrix1, distCoeff1;
	Mat R, T, R1, R2, P1, P2, Q;
	Mat map11, map12, map21, map22;
	Size imgSize;
	Rect rect1, rect2;
};

