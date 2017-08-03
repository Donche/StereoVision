#pragma once

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgcodecs.hpp>

using cv::Mat;
using cv::Mat_;
using cv::Size;
using cv::Rect;
using std::vector;
using cv::Point2f;
using cv::DMatch;

class StereoVision
{
public:
	StereoVision();
	~StereoVision();

	enum StereoType { PHOTO_BM, PHOTO_SGBM, VISION_BM, VISION_SGBM, 
		PHOTO_SURF, VISION_SURF, VISION_ORB };
	enum MatchType { KNN_MATCH, BRUTE_FORCE};

	bool runStereoVision(StereoType stereoType, MatchType matchType = KNN_MATCH);

private:

	//BM
	bool initBMSV(Mat& map11, Mat& map12, Mat& map21, Mat& map22, Mat& Q);
	bool runBMStereoVision(StereoType stereoType);
	bool runBMStereoPhoto(StereoType stereoType);
	void fixDisparity(Mat_<float> & disp, int numberOfDisparities);
	void BM23D(Mat& disparity32F, Mat& Q);

	//Key Points
	bool initFeatureSV(Mat& K);
	bool runFeatureStereoVision(StereoType stereoType, MatchType matchType);
	bool runFeatureStereoPhoto(StereoType stereoType, MatchType matchType);

	vector<DMatch> ratioTest(vector<vector<DMatch>>& rawMatches, double ratioDist, int ratioMinDist);
	void symmetryTest(const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2, std::vector<cv::DMatch>& symMatches);
	bool calRnT(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T);
	bool keyPoint23D(Mat& K, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure);
};

