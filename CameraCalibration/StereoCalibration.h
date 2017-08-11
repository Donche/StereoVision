#pragma once

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using std::vector;
using cv::Mat;
using cv::Size;
using cv::Point2f;
using cv::Point3f;
using cv::Scalar;

class StereoCalibration
{
public:
	StereoCalibration();

	~StereoCalibration() {

	}
	enum Pattern { NOT_EXISTING, CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };
	bool runStereoCalibration(Pattern pat);


private:
	 Mat GetImage(int camNum, int i);
	void calcBoardCornerPositions( vector< Point3f>& corners);
	double computeReprojectionErrors(const  vector< vector< Point3f> >& objectPoints,
		const  vector< Mat>& rvecs, const  vector< Mat>& tvecs,
		 vector<float>& perViewErrors, bool camNum);
	bool runCalibration( vector< vector< Point3f> >& objectPoints,
		 vector< Mat>& rvecs,  vector< Mat>& tvecs,
		 vector<float>& reprojErrs, double& totalAvgErr, bool camNum);
	void saveCameraParams( const  vector< Mat>& rvecs, const  vector< Mat>& tvecs,
		const  vector<float>& reprojErrs, double totalAvgErr, bool num);
	bool runCalibrationAndSave(vector<vector<Point3f> >& objectPoints, bool num);


	vector<vector<vector<Point2f> > > imagePoints;
	vector<vector<Point3f> > objectPoints;
	vector<Mat> cameraMatrix, distCoeffs;
	vector<Mat> map;
	Mat Q;
	Size imgSize;
	Size boardSize;
	float squareSize;
	clock_t prevTimestamp = 0;
	const Scalar RED, GREEN;
	const char ESC_KEY = 27;
	vector<Mat> view;
};

