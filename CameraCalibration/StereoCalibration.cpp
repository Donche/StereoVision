#include "StereoCalibration.h"

#define WIDTH 7
#define HEIGHT 7
#define SQUARE_SIZE 0.027
#define OUTPUT_FILENAME "result.yml"
#define STEREO_OUTPUT_FILENAME "stereo_result.yml"

using namespace std;
using namespace cv;


StereoCalibration::StereoCalibration():RED(Scalar(0, 0, 255)), GREEN(Scalar(0, 255, 0)) {
	imagePoints.resize(2);
	objectPoints.resize(1);
	cameraMatrix.resize(2);
	map.resize(4);
	distCoeffs.resize(2);
	view.resize(2);
	boardSize.width = WIDTH;
	boardSize.height = HEIGHT;
	squareSize = SQUARE_SIZE;
}



bool StereoCalibration::runStereoCalibration(Pattern pat)
{
	int imageNumber = 0;
	cout << "input the amount of images" << endl;
	cin >> imageNumber;
	if (cin.fail()) {
		std::cin.clear();
		std::cin.ignore();
		cout << "invalid input" << endl;
		return false;
	}

	for (int i = 0; i < imageNumber; ++i) {
		view[0] = GetImage(0, i);
		view[1] = GetImage(1, i);
		if (view[0].empty() || view[1].empty()) {
			cout << "image " << i << "invalid, continue to the next image" << endl;
			continue;
		}

		imgSize = view[0].size();  // Format input image.
		vector<vector<Point2f>> pointBuf(2);
		vector<bool> found(2);
		if (pat == CHESSBOARD) {
			found[0] = findChessboardCorners(view[0], Size(WIDTH, HEIGHT), pointBuf[0],
				CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
			found[1] = findChessboardCorners(view[1], Size(WIDTH, HEIGHT), pointBuf[1],
				CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
		}
		else if (pat == CIRCLES_GRID) {
			found[0] = findCirclesGrid(view[0], Size(WIDTH, HEIGHT), pointBuf[0]);
			found[1] = findCirclesGrid(view[1], Size(WIDTH, HEIGHT), pointBuf[1]);
		}
		cout << i << "--> Found : " << found[0] << " " << found[1] << endl;

		if (found[0] && found[1])
		{
			if (pat == CHESSBOARD) {
				vector<Mat> viewGray(2);
				cvtColor(view[0], viewGray[0], COLOR_BGR2GRAY);
				cornerSubPix(viewGray[0], pointBuf[0], Size(11, 11),
					Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

				cvtColor(view[1], viewGray[1], COLOR_BGR2GRAY);
				cornerSubPix(viewGray[1], pointBuf[1], Size(11, 11),
					Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			}
			drawChessboardCorners(view[0], Size(WIDTH, HEIGHT), Mat(pointBuf[0]), found[0]);
			imagePoints[0].push_back(pointBuf[0]);
			drawChessboardCorners(view[1], Size(WIDTH, HEIGHT), Mat(pointBuf[1]), found[1]);
			imagePoints[1].push_back(pointBuf[1]);
		}
	}
	//loop over
	if (imagePoints[0].size() == 0 || imagePoints[1].size() == 0)
	{
		cout << "Calibration failed..." << endl;
		return 0;
	}

	if (runCalibrationAndSave(objectPoints,0))
	{
		Mat temp = view[0].clone();
		undistort(temp, view[0], cameraMatrix[0], distCoeffs[0]);
	}

	if (runCalibrationAndSave(objectPoints,1))
	{
		Mat temp = view[1].clone();
		undistort(temp, view[1], cameraMatrix[1], distCoeffs[1]);
	}

#pragma region UNUSED

	//Mat rview, map1, map2;
	//initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], Mat(),
	//	getOptimalNewCameraMatrix(cameraMatrix[0], distCoeffs[0], imageSize, 1, imageSize, 0),
	//	imageSize, CV_16SC2, map1, map2);

	//initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], Mat(),
	//	getOptimalNewCameraMatrix(cameraMatrix[0], distCoeffs[0], imageSize, 1, imageSize, 0),
	//	imageSize, CV_16SC2, map1, map2);

	//for (int i = 0; i < IMAGENUMBER; i++)
	//{
	//	view = GetImage(0, i);
	//	if (view.empty())
	//		continue;
	//	remap(view, rview, map1, map2, INTER_LINEAR);
	//	imshow("Image View", rview);
	//	char c = (char)waitKey();
	//	if (c == ESC_KEY || c == 'q' || c == 'Q')
	//		break;
	//}

#pragma endregion

	Mat R, T, E, F, R1, R2, P1, P2;
	if (imagePoints[0].size() == 0 || imagePoints[1].size() == 0) return 0;
	stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1], cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1], imgSize, R, T, E, F);

	stereoRectify(cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imgSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY);

	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imgSize, CV_16SC2, map[0], map[1]);
	initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imgSize, CV_16SC2, map[2], map[3]);


	FileStorage fs(STEREO_OUTPUT_FILENAME, FileStorage::WRITE);
	if (fs.isOpened()) {
		fs << "Camera_Matrix" << cameraMatrix[0];
		fs << "Distortion_Coefficients" << distCoeffs[0];
		fs << "F" << F;
		fs << "E" << E;
		fs << "R" << R;
		fs << "T" << T;
		fs << "Q" << Q;
		fs << "map11" << map[0];
		fs << "map12" << map[1]; 
		fs << "map21" << map[2]; 
		fs << "map22" << map[3]; 
		fs.release();
	}
	else {
		cout << "can not open the file" << endl;
		return 0;
	}
	return 1;
}


Mat StereoCalibration::GetImage(int camNum, int i) {
	Mat res;
	string camCode = (camNum == 0) ? "_0" : "_1";
	string imageName = "CalibrationImages\\image" + to_string(i) + camCode + ".jpg";
	res = imread(imageName, CV_LOAD_IMAGE_COLOR);
	return res;
}

void StereoCalibration::calcBoardCornerPositions( vector<Point3f>& corners)
{
	corners.clear();

	for (int i = 0; i < boardSize.height; ++i)
		for (int j = 0; j < boardSize.width; ++j)
			corners.push_back(Point3f(float(j*squareSize), float(i*squareSize), 0));
}

double StereoCalibration::computeReprojectionErrors(const  vector< vector< Point3f> >& objectPoints,
	const  vector< Mat>& rvecs, const  vector< Mat>& tvecs,
	vector<float>& perViewErrors, bool camNum)
{
	vector<Point2f> imagePoints2;
	int i, totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for (i = 0; i < (int)objectPoints.size(); ++i)
	{
		projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix[camNum],
			distCoeffs[camNum], imagePoints2);
		err = norm(Mat(imagePoints[camNum][i]), Mat(imagePoints2), CV_L2);

		int n = (int)objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err*err / n);
		totalErr += err*err;
		totalPoints += n;
	}

	return std::sqrt(totalErr / totalPoints);
}

bool StereoCalibration::runCalibration(vector< vector< Point3f> >& objectPoints,
	vector< Mat>& rvecs, vector< Mat>& tvecs,
	vector<float>& reprojErrs, double& totalAvgErr, bool camNum)
{

	cameraMatrix[camNum] = Mat::eye(3, 3, CV_64F);
	cameraMatrix[camNum].at<double>(0, 0) = 1.0;

	distCoeffs[camNum] = Mat::zeros(8, 1, CV_64F);

	calcBoardCornerPositions(objectPoints[0]);

	objectPoints.resize(imagePoints[camNum].size(), objectPoints[0]);

	//Find intrinsic and extrinsic camera parameters
	int flag = 0;
	double rms = calibrateCamera(objectPoints, imagePoints[camNum], imgSize, cameraMatrix[camNum],
		distCoeffs[camNum], rvecs, tvecs, flag | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5);

	cout << "Re-projection error reported by calibrateCamera: " << rms << endl;

	bool ok = checkRange(cameraMatrix[camNum]) && checkRange(distCoeffs[camNum]);

	totalAvgErr = computeReprojectionErrors(objectPoints,
		rvecs, tvecs, reprojErrs, camNum);

	return ok;
}

void StereoCalibration::saveCameraParams(const  vector< Mat>& rvecs, const  vector< Mat>& tvecs,
	const  vector<float>& reprojErrs, double totalAvgErr, bool camNum)
{
	FileStorage fs(to_string(camNum) + "_" + OUTPUT_FILENAME, FileStorage::WRITE);

	time_t tm;
	time(&tm);
	struct tm *t2 = localtime(&tm);
	char buf[1024];
	strftime(buf, sizeof(buf) - 1, "%c", t2);

	fs << "calibration_Time" << buf;

	if (!rvecs.empty() || !reprojErrs.empty())
		fs << "nrOfFrames" << (int)std::max(rvecs.size(), reprojErrs.size());
	fs << "image_Width" << imgSize.width;
	fs << "image_Height" << imgSize.height;
	fs << "board_Width" << WIDTH;
	fs << "board_Height" << HEIGHT;
	fs << "square_Size" << SQUARE_SIZE;

	fs << "Camera_Matrix" << cameraMatrix[camNum];
	fs << "Distortion_Coefficients" << distCoeffs[camNum];

	fs << "Avg_Reprojection_Error" << totalAvgErr;
	if (!reprojErrs.empty())
		fs << "Per_View_Reprojection_Errors" << Mat(reprojErrs);

	if (!rvecs.empty() && !tvecs.empty())
	{
		CV_Assert(rvecs[0].type() == tvecs[0].type());
		Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
		for (int i = 0; i < (int)rvecs.size(); i++)
		{
			Mat r = bigmat(Range(i, i + 1), Range(0, 3));
			Mat t = bigmat(Range(i, i + 1), Range(3, 6));

			CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
			CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
			//*.t() is MatExpr (not Mat) so we can use assignment operator
			r = rvecs[i].t();
			t = tvecs[i].t();
		}
		cvWriteComment(*fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0);
		fs << "Extrinsic_Parameters" << bigmat;
	}

	if (!imagePoints[camNum].empty())
	{
		Mat imagePtMat((int)imagePoints[camNum].size(), (int)imagePoints[camNum][0].size(), CV_32FC2);
		for (int i = 0; i < (int)imagePoints[camNum].size(); i++)
		{
			Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
			Mat imgpti(imagePoints[camNum][i]);
			imgpti.copyTo(r);
		}
		fs << "Image_points" << imagePtMat;
	}
}

bool StereoCalibration::runCalibrationAndSave(vector<vector<Point3f> >& objectPoints, bool camNum)
{
	vector<Mat> rvecs, tvecs;
	vector<float> reprojErrs;
	double totalAvgErr = 0;

	bool ok = runCalibration(objectPoints,rvecs, tvecs, reprojErrs, totalAvgErr, camNum);
	cout << (ok ? "Calibration succeeded" : "Calibration failed")
		<< ". avg re projection error = " << totalAvgErr << endl;

	if (ok)
		saveCameraParams(rvecs, tvecs, reprojErrs,totalAvgErr, camNum);
	return ok;
}