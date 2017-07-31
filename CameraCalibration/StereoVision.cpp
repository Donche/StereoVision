#include "StereoVision.h"

using namespace std;
using namespace cv;

#define IMAGE_HEIGHT 777
#define IMAGE_WIDTH 1000
#define DISP_IMG_HEIGHT 400
#define DISP_IMG_WIDTH 500

StereoVision::StereoVision()
{

}

StereoVision::~StereoVision()
{
}

bool StereoVision::initSV() {
	FileStorage stereofs;
	FileStorage fs0;
	FileStorage fs1;
	if (!stereofs.open("stereo_result.yml", FileStorage::READ) ||
		!fs0.open("0_result.yml", FileStorage::READ) ||
		!fs1.open("1_result.yml", FileStorage::READ)) {
		cout << "can not find configuration files" << endl;
		return 0;
	}

	fs0["Camera_Matrix"] >> Camera_Matrix0;
	fs0["Distortion_Coefficients"] >> distCoeff0;
	fs1["Camera_Matrix"] >> Camera_Matrix1;
	fs1["Distortion_Coefficients"] >> distCoeff1;
	fs0["image_Height"] >> imgSize.height;
	fs0["image_Width"] >> imgSize.width;
	stereofs["R"] >> R;
	stereofs["T"] >> T;
	stereoRectify(Camera_Matrix0, distCoeff0, Camera_Matrix1, distCoeff1, imgSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY);

	initUndistortRectifyMap(Camera_Matrix0, distCoeff0, R1, P1, imgSize, CV_16SC2, map11, map12);
	initUndistortRectifyMap(Camera_Matrix1, distCoeff1, R2, P2, imgSize, CV_16SC2, map21, map22);

	rect1 = Rect(0, 0, DISP_IMG_WIDTH, DISP_IMG_HEIGHT);
	rect2 = Rect(DISP_IMG_WIDTH, 0, DISP_IMG_WIDTH, DISP_IMG_HEIGHT);

	return 1;
}

void StereoVision::runVision(){
	cout << "1.BM Real Time Vision \t 2.BM Photo Procession \t 3.SGBM Real Time Vision\n "
		<<"4.SIFT Photo Procession\t 5.SIFT Real Time Vision\t 0.Return" << endl;
	char inp;
	cin >> inp;
	if (inp == '1')	runStereoVisionBM();
	else if (inp == '2') runPhotoStereoVisionBM();
	else if (inp == '3') runStereoVisionSGBM();
	else if (inp == '4') runStereoPhotoSIFT();
	else if (inp == '5') runStereoVisionSIFT();
	else if (inp == '0') return;
	else {
	cout << "Seriously" << endl;
}
}

bool StereoVision::runStereoVisionBM() {

	VideoCapture cam1(1);
	VideoCapture cam0(0);
	cam0.set(CV_CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH);
	cam0.set(CV_CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT);
	cam1.set(CV_CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH);
	cam1.set(CV_CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT);
	Mat imgLeft, imgRight, imgLeft_tmp, imgRight_tmp, imgTotal;
	while (1)
	{
		cam0 >> imgLeft;
		cam1 >> imgRight;
		cvtColor(imgLeft, imgLeft, CV_BGR2GRAY);
		cvtColor(imgRight, imgRight, CV_BGR2GRAY);
		//imshow("o_view0", imgLeft);
		//imshow("o_view1", imgRight);
		remap(imgLeft, imgLeft_tmp, map11, map12, INTER_LINEAR);
		remap(imgRight, imgRight_tmp, map21, map22, INTER_LINEAR);
		imgLeft = imgLeft_tmp;
		imgRight = imgRight_tmp;

		imgTotal.create(Size(DISP_IMG_WIDTH * 2, DISP_IMG_HEIGHT), CV_8UC1);
		resize(imgLeft, imgTotal(rect1), Size(DISP_IMG_WIDTH, DISP_IMG_HEIGHT));
		resize(imgRight, imgTotal(rect2), Size(DISP_IMG_WIDTH, DISP_IMG_HEIGHT));

		//imshow("view0", imgLeft);
		//imshow("view1", imgRight);
		imshow("Img", imgTotal);

		Mat imgDisparity32F = Mat(imgLeft.rows, imgLeft.cols, CV_32F);
		Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

		if (imgLeft.empty() || imgRight.empty())
		{
			std::cout << " --(!) Error reading images " << std::endl;
			waitKey(1000);
			continue;
		}

		//-- 2. Call the constructor for StereoBM
		int ndisparities = 16 * 3;   /**< Range of disparity */
		int SADWindowSize = 11; /**< Size of the block window. Must be odd */
		
		StereoBM bm(StereoBM::BASIC_PRESET,ndisparities,SADWindowSize);
		bm.state->textureThreshold = 20;
		bm.state->speckleRange = 32;
		bm.state->preFilterType = CV_STEREO_BM_NORMALIZED_RESPONSE;
		bm.state->preFilterSize = 9;
		bm.state->preFilterCap = 3;
		bm.state->minDisparity = 0;
		bm.state->uniquenessRatio = 5;
		bm.state->speckleRange = 32;
		bm.state->speckleWindowSize = 100;

		//bm->setBlockSize(21);

		//-- 3. Calculate the disparity image

		bm(imgLeft, imgRight, imgDisparity32F, CV_32F);

		//-- Check its extreme values
		double minVal; double maxVal;
		minMaxLoc(imgDisparity32F, &minVal, &maxVal);

		//printf("Min disp: %f Max value: %f \n", minVal, maxVal);

		//-- 4. Display it as a CV_8UC1 image
		imgDisparity32F.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal));

		namedWindow("Disparity", WINDOW_NORMAL);
		imshow("Disparity", imgDisparity8U);


		waitKey(300);

	}
	return 1;
}

bool StereoVision::runPhotoStereoVisionBM() {
	Mat imgLeft = imread("CalibrationStaticImages\\1.jpg");
	Mat imgRight = imread("CalibrationStaticImages\\2.jpg");
	cvtColor(imgLeft, imgLeft, CV_BGR2GRAY);
	cvtColor(imgRight, imgRight, CV_BGR2GRAY);

	Mat imgDisparity16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
	Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

	int ndisparities = 16 * 3;   /**< Range of disparity */
	int SADWindowSize = 19; /**< Size of the block window. Must be odd */

	StereoBM bm(StereoBM::BASIC_PRESET, ndisparities, SADWindowSize);
	//bm->setPreFilterType(CV_STEREO_BM_NORMALIZED_RESPONSE);
	//bm->setPreFilterSize(9);
	//bm->setPreFilterCap(31);
	//bm->setBlockSize(21);
	//bm->setMinDisparity(0);
	//bm->setNumDisparities(ndisparities);
	//bm->setTextureThreshold(10);
	//bm->setUniquenessRatio(15);
	//bm->setSpeckleWindowSize(100);
	//bm->setSpeckleRange(32);

	//-- 3. Calculate the disparity image

	bm(imgLeft, imgRight, imgDisparity16S, CV_16S);

	//-- Check its extreme values
	double minVal; double maxVal;
	minMaxLoc(imgDisparity16S, &minVal, &maxVal);

	//printf("Min disp: %f Max value: %f \n", minVal, maxVal);

	//-- 4. Display it as a CV_8UC1 image
	imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal));
	cv::normalize(imgDisparity8U, imgDisparity8U, 0, 255, CV_MINMAX, CV_8UC1);

	namedWindow("Disparity", WINDOW_NORMAL);
	imshow("Disparity", imgDisparity8U);
	waitKey(100);

	return 1;
}

bool StereoVision::runStereoVisionSGBM() {

	VideoCapture cam1(1);
	VideoCapture cam0(0);
	cam0.set(CV_CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH);
	cam0.set(CV_CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT);
	cam1.set(CV_CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH);
	cam1.set(CV_CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT);
	Mat imgLeft, imgRight, imgLeft_tmp, imgRight_tmp, imgTotal;
	while (1)
	{
		cam0 >> imgLeft;
		cam1 >> imgRight;
		//cvtColor(imgLeft, imgLeft, CV_BGR2GRAY);
		//cvtColor(imgRight, imgRight, CV_BGR2GRAY);
		//imshow("o_view0", imgLeft);
		//imshow("o_view1", imgRight);
		remap(imgLeft, imgLeft_tmp, map11, map12, INTER_LINEAR);
		remap(imgRight, imgRight_tmp, map21, map22, INTER_LINEAR);
		imgLeft = imgLeft_tmp;
		imgRight = imgRight_tmp;

		imgTotal.create(Size(DISP_IMG_WIDTH * 2, DISP_IMG_HEIGHT), CV_8UC3);
		resize(imgLeft, imgTotal(rect1), Size(DISP_IMG_WIDTH, DISP_IMG_HEIGHT));
		resize(imgRight, imgTotal(rect2), Size(DISP_IMG_WIDTH, DISP_IMG_HEIGHT));

		//imshow("view0", imgLeft);
		//imshow("view1", imgRight);
		imshow("Img", imgTotal);

		Mat imgDisparity16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
		Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

		if (imgLeft.empty() || imgRight.empty())
		{
			std::cout << " --(!) Error reading images " << std::endl;
			waitKey(1000);
			continue;
		}

		//-- 2. Call the constructor for StereoBM
		int ndisparities = 16 * 3;   /**< Range of disparity */
		int SADWindowSize = 11; /**< Size of the block window. Must be odd */
		int minDisparity = 0;
		StereoSGBM sgbm(minDisparity, ndisparities, SADWindowSize);
		sgbm.uniquenessRatio = 5;
		sgbm.preFilterCap = 3;
		sgbm.speckleRange = 32;
		sgbm.speckleWindowSize = 100;
		
		int cn = imgLeft.channels();

		sgbm.P1 =  8 * cn * SADWindowSize * SADWindowSize;
		sgbm.P2 = 32 * cn * SADWindowSize * SADWindowSize;

		//bm->setBlockSize(21);

		//-- 3. Calculate the disparity image

		sgbm(imgLeft, imgRight, imgDisparity16S);

		//-- Check its extreme values
		double minVal; double maxVal;
		minMaxLoc(imgDisparity16S, &minVal, &maxVal);

		//printf("Min disp: %f Max value: %f \n", minVal, maxVal);

		//-- 4. Display it as a CV_8UC1 image
		imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal));

		namedWindow("Disparity", WINDOW_NORMAL);
		imshow("Disparity", imgDisparity8U);


		waitKey(300);

	}
	return 1;
}

bool StereoVision::runStereoPhotoSIFT() {

	vector<vector<KeyPoint>> key_points_for_all;
	vector<Mat> descriptor_for_all;
	vector<vector<Vec3b>> colors_for_all;

	//VideoCapture cam1(1);
	//VideoCapture cam0(0);
	//cam0.set(CV_CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH);
	//cam0.set(CV_CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT);
	//cam1.set(CV_CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH);
	//cam1.set(CV_CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT);
	//Mat imgLeft, imgRight, imgLeft_tmp, imgRight_tmp, imgTotal;
	//cam0 >> imgLeft;
	//cam1 >> imgRight;

	Mat imgLeft = imread("CalibrationStaticImages\\1.jpg");
	Mat imgRight = imread("CalibrationStaticImages\\2.jpg");
	if (imgLeft.empty() || imgRight.empty()) return 0;

	cvtColor(imgLeft, imgLeft, CV_BGR2GRAY);
	cvtColor(imgRight, imgRight, CV_BGR2GRAY);

	//读取图像，获取图像特征点
	SiftFeatureDetector detector;
	SiftDescriptorExtractor extractor;
	vector<KeyPoint> key_points_l, key_points_r;
	Mat descriptor_l, descriptor_r;
	detector.detect(imgLeft, key_points_l);
	detector.detect(imgRight, key_points_r);

	//画出特征点
	//Mat imageOutput1;
	//Mat imageOutput2;
	//drawKeypoints(imgLeft, key_points_l, imageOutput1, Scalar(0, 255, 0));
	//drawKeypoints(imgRight, key_points_r, imageOutput2, Scalar(0, 255, 0));

	//计算描述子
	extractor.compute(imgLeft, key_points_l, descriptor_l);
	extractor.compute(imgRight, key_points_r, descriptor_r);

#pragma region KnnMatch
	//Match Features
	BFMatcher matcher(NORM_L2);
	vector<vector<DMatch>> knn_matches;
	matcher.knnMatch(descriptor_l, descriptor_r, knn_matches, 2);

	//获取满足Ratio Test的最小匹配的距离
	float min_dist = FLT_MAX;
	for (int r = 0; r < knn_matches.size(); ++r)
	{
		//Ratio Test
		if (knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance)
			continue;

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist) min_dist = dist;
	}

	vector<DMatch> matches;
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		//排除不满足Ratio Test的点和匹配距离过大的点
		if (
			knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance ||
			knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
			)
			continue;

		//保存匹配点
		matches.push_back(knn_matches[r][0]);
	}

#pragma endregion

#pragma region BruteForceMatch
	////暴力匹配
	//BruteForceMatcher<L2<float>> matcher;
	//vector<DMatch> matches;
	//matcher.match(descriptor_l, descriptor_r, matches);

	////挑选匹配的最好的前100个
	//nth_element(matches.begin(), matches.begin() + 99, matches.end());
	//matches.erase(matches.begin() + 99, matches.end());

#pragma endregion

	Mat outImg;

drawMatches(imgLeft, key_points_l, imgRight, key_points_r, matches, outImg);
imshow("outImg", outImg);

waitKey(100);
getchar();
getchar();
return 1;
}

bool StereoVision::runStereoVisionSIFT() {

	vector<vector<KeyPoint>> key_points_for_all;
	vector<Mat> descriptor_for_all;
	vector<vector<Vec3b>> colors_for_all;

	VideoCapture cam1(1);
	VideoCapture cam0(0);
	if (!cam1.isOpened() || !cam0.isOpened()) return false;
	cam0.set(CV_CAP_PROP_FRAME_WIDTH, DISP_IMG_WIDTH);
	cam0.set(CV_CAP_PROP_FRAME_HEIGHT, DISP_IMG_HEIGHT);
	cam1.set(CV_CAP_PROP_FRAME_WIDTH, DISP_IMG_WIDTH);
	cam1.set(CV_CAP_PROP_FRAME_HEIGHT, DISP_IMG_HEIGHT);
	Mat imgLeft, imgRight, outImg;;
	SiftFeatureDetector detector;
	SiftDescriptorExtractor extractor;
	vector<KeyPoint> key_points_l, key_points_r;
	Mat descriptor_l, descriptor_r;
	BFMatcher matcher(NORM_L2);
	vector<vector<DMatch>> knn_matches;
	vector<DMatch> matches;

	////BruteForceMatcher
	//BruteForceMatcher<L2<float>> matcher;

	while (1) {

		key_points_l.clear();
		key_points_r.clear();

		cam0 >> imgLeft;
		cam1 >> imgRight;

		if (imgLeft.empty() || imgRight.empty()) continue;

		cvtColor(imgLeft, imgLeft, CV_BGR2GRAY);
		cvtColor(imgRight, imgRight, CV_BGR2GRAY);

		//获取图像特征点
		detector.detect(imgLeft, key_points_l);
		detector.detect(imgRight, key_points_r);

		//画出特征点
		//Mat imageOutput1;
		//Mat imageOutput2;
		//drawKeypoints(imgLeft, key_points_l, imageOutput1, Scalar(0, 255, 0));
		//drawKeypoints(imgRight, key_points_r, imageOutput2, Scalar(0, 255, 0));

		//计算描述子
		extractor.compute(imgLeft, key_points_l, descriptor_l);
		extractor.compute(imgRight, key_points_r, descriptor_r);

		#pragma region KnnMatch
				//knn matches
				vector<DMatch> matches;
		
				//Match Features
				matcher.knnMatch(descriptor_l, descriptor_r, knn_matches, 2);
		
				//获取满足Ratio Test的最小匹配的距离
				float min_dist = FLT_MAX;
				for (int r = 0; r < knn_matches.size(); ++r)
				{
					//Ratio Test
					if (knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance)
						continue;
		
					float dist = knn_matches[r][0].distance;
					if (dist < min_dist) min_dist = dist;
				}
		
				for (size_t r = 0; r < knn_matches.size(); ++r)
				{
					//排除不满足Ratio Test的点和匹配距离过大的点
					if (
						knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance ||
						knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
						)
						continue;
		
					//保存匹配点
					matches.push_back(knn_matches[r][0]);
				}
		
		#pragma endregion

		//#pragma region BruteForceMatch
		//		//暴力匹配
		//		matches.clear();
		//		matcher.match(descriptor_l, descriptor_r, matches);
		//
		//		//挑选匹配的最好的前100个
		//		if (matches.size() > 100) {
		//			nth_element(matches.begin(), matches.begin() + 99, matches.end());
		//			matches.erase(matches.begin() + 99, matches.end());
		//		}
		//
		//#pragma endregion

		drawMatches(imgLeft, key_points_l, imgRight, key_points_r, matches, outImg);
		imshow("outImg", outImg);

		waitKey(100);
	}
	return 1;
}

void StereoVision::BM23D(Mat& disparity32F) {
	Mat_<Vec3f> XYZ(disparity32F.rows, disparity32F.cols);   // Output point cloud
	Mat_<float> vec_tmp(4, 1);
	for (int y = 0; y<disparity32F.rows; ++y) {
		for (int x = 0; x<disparity32F.cols; ++x) {
			vec_tmp(0) = x; 
			vec_tmp(1) = y; 
			vec_tmp(2) = disparity32F.at<float>(y,x) ; 
			vec_tmp(3) = 1;

			vec_tmp = Q*vec_tmp;
			vec_tmp /= vec_tmp(3);

			cv::Vec3f &point = XYZ.at<cv::Vec3f>(y, x);
			point[0] = vec_tmp(0);
			point[1] = vec_tmp(1);
			point[2] = vec_tmp(2);
		}
	}

}