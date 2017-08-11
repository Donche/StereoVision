#include "StereoVision.h"

using namespace std;
using namespace cv;

#define IMAGE_HEIGHT 768
#define IMAGE_WIDTH 1024
#define DISP_IMG_HEIGHT 350
#define DISP_IMG_WIDTH 400
#define OPENGL_DISP false
#define BM_POST_FILTER true

StereoVision::StereoVision()
{
}

StereoVision::~StereoVision()
{
}

bool StereoVision::runStereoVision(StereoType stereoType, MatchType matchType) {
	if (stereoType < 4) {
		if (stereoType < 2) runBMStereoPhoto(stereoType);
		else runBMStereoVision(stereoType);
	}
	else {
		if(stereoType > 4) runFeatureStereoVision(stereoType, matchType);
		else runFeatureStereoPhoto(stereoType, matchType);
	}
	return 1;
}

/**------------------- Block Match ----------------------**/

bool StereoVision::initBMSV(Mat& map11, Mat& map12, Mat& map21, Mat& map22, Mat& Q) {
	Mat Camera_Matrix0, distCoeff0, Camera_Matrix1, distCoeff1;
	Mat R, T, R1, R2, P1, P2;
	Size imgSize;

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
	stereofs["map11"] >> map11;
	stereofs["map12"] >> map12;
	stereofs["map21"] >> map21;
	stereofs["map22"] >> map22;
	stereofs["Q"] >> Q;
	//stereoRectify(Camera_Matrix0, distCoeff0, Camera_Matrix1, distCoeff1, imgSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY);

	//initUndistortRectifyMap(Camera_Matrix0, distCoeff0, R1, P1, imgSize, CV_16SC2, map11, map12);
	//initUndistortRectifyMap(Camera_Matrix1, distCoeff1, R2, P2, imgSize, CV_16SC2, map21, map22);

	return 1;
}

bool StereoVision::runBMStereoVision(StereoType stereoType)  {

	VideoCapture cam1(1);
	VideoCapture cam0(0);
	if (!cam1.isOpened() || !cam0.isOpened()) {
		cout << "can not open camera" << endl;
		return 0;
	}
	cam0.set(CV_CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH);
	cam0.set(CV_CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT);
	cam1.set(CV_CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH);
	cam1.set(CV_CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT);

	Mat map11, map12, map21, map22, Q, channel[3];
	if (!initBMSV(map11, map12, map21, map22, Q)) return 0;

	Rect rect1, rect2, rect3, rect4;
	rect1 = Rect(0, 0, DISP_IMG_WIDTH, DISP_IMG_HEIGHT);
	rect2 = Rect(DISP_IMG_WIDTH, 0, DISP_IMG_WIDTH, DISP_IMG_HEIGHT);
	rect3 = Rect(0, DISP_IMG_HEIGHT, DISP_IMG_WIDTH, DISP_IMG_HEIGHT);
	rect4 = Rect(DISP_IMG_WIDTH, DISP_IMG_HEIGHT, DISP_IMG_WIDTH, DISP_IMG_HEIGHT);

	Mat imgLeft, imgRight, imgLeft_tmp, imgRight_tmp, imgTotal;
	Mat texture;
	imgTotal.create(Size(DISP_IMG_WIDTH * 2, DISP_IMG_HEIGHT * 2), CV_8UC1);


	int preFilterSize = 10; /*Musr be odd*/
	int preFilterCap = 36;

	int ndisparities = 5;   /**< Range of disparity */
	int SADWindowSize = 4; /**< Size of the block window. Must be odd */

	int textureThreshold = 507;
	int uniquenessRatio = 4;
	int speckleWindowSize = 6;
	int speckleRange = 14;

	int lambda = 3;
	int sigma = 3;

	cvNamedWindow("Parameters Adjustment");
	cvMoveWindow("Parameters Adjustment", 10, 5);
	cvResizeWindow("Parameters Adjustment", 450, 450);
	createTrackbar("preFilterSize", "Parameters Adjustment", &preFilterSize, 10);
	//createTrackbar("blockSize", "Parameters Adjustment", &blockSize, 40);
	createTrackbar("preFilterCap", "Parameters Adjustment", &preFilterCap, 61);

	createTrackbar("ndisparities", "Parameters Adjustment", &ndisparities, 7);
	createTrackbar("SADWindowSize", "Parameters Adjustment", &SADWindowSize, 12);

	createTrackbar("textureThreshold", "Parameters Adjustment", &textureThreshold, 507);
	createTrackbar("uniquenessRatio", "Parameters Adjustment", &uniquenessRatio, 10);
	createTrackbar("speckleWindowSize", "Parameters Adjustment", &speckleWindowSize, 150);
	createTrackbar("speckleRange", "Parameters Adjustment", &speckleRange, 50);

	createTrackbar("lambda", "Parameters Adjustment", &lambda, 150);
	createTrackbar("sigma", "Parameters Adjustment", &sigma, 150);

	//createTrackbar("an", "Parameters Adjustment", &an, 10);

	//---OpenGL display initiate
	if (OPENGL_DISP)	
		initDisp(IMAGE_HEIGHT, IMAGE_WIDTH);

	while (1)
	{
		cam1 >> imgLeft;
		cam0 >> imgRight;
		if (imgLeft.empty() || imgRight.empty())
		{
			std::cout << " --(!) Error reading images " << std::endl;
			cv::waitKey(1000);
			continue;
		}

		flip(imgLeft, imgLeft, 0);
		flip(imgRight, imgRight, 0);
		split(imgLeft, channel);
		remap(channel[0], channel[0], map21, map22, INTER_LINEAR);
		remap(channel[1], channel[1], map21, map22, INTER_LINEAR);
		remap(channel[2], channel[2], map21, map22, INTER_LINEAR);
		merge(channel, 3, texture);

		cvtColor(imgLeft, imgLeft, CV_BGR2GRAY);
		cvtColor(imgRight, imgRight, CV_BGR2GRAY);

		GaussianBlur(imgLeft, imgLeft, Size(1, 1), 20, 0);
		GaussianBlur(imgRight, imgRight, Size(1, 1), 20, 0);
		//equalizeHist(imgLeft, imgLeft);
		//equalizeHist(imgRight, imgRight);
		//equalizeHist(imgLeft, imgLeft);
		//equalizeHist(imgRight, imgRight);

		remap(imgLeft, imgLeft_tmp, map21, map22, INTER_LINEAR);
		remap(imgRight, imgRight_tmp, map11, map12, INTER_LINEAR);
		imgLeft = imgLeft_tmp;
		imgRight = imgRight_tmp;

		resize(imgLeft, imgTotal(rect1), Size(DISP_IMG_WIDTH, DISP_IMG_HEIGHT));
		resize(imgRight, imgTotal(rect2), Size(DISP_IMG_WIDTH, DISP_IMG_HEIGHT));

		Mat imgDisparity16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
		Mat imgDisparity32F = Mat(imgLeft.rows, imgLeft.cols, CV_32F);
		Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);
		double minVal; double maxVal;

		//-- 2. Call the constructor for StereoBM
		if (stereoType = VISION_BM) {

			Ptr<StereoBM> bm = StereoBM::create(ndisparities * 16, SADWindowSize * 2 +5);
			//bm->setPreFilterType(CV_STEREO_BM_NORMALIZED_RESPONSE);
			//bm->setPreFilterSize(9);
			bm->setPreFilterCap(preFilterCap + 1);
			bm->setPreFilterSize(preFilterSize * 2 + 5);
			//bm->setBlockSize(blockSize);
			bm->setMinDisparity(-16);
			bm->setTextureThreshold(textureThreshold);
			bm->setUniquenessRatio(uniquenessRatio);
			bm->setSpeckleWindowSize(speckleWindowSize);
			bm->setSpeckleRange(speckleRange);
			bm->setDisp12MaxDiff(1);

			//-- 3. Calculate the disparity image
			if (BM_POST_FILTER) {
				Mat imgDisparity16SLeft = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
				Mat imgDisparity16SRight = Mat(imgLeft.rows, imgLeft.cols, CV_16S);

				bm->compute(imgLeft, imgRight, imgDisparity16SLeft);
				bm->compute(imgRight, imgLeft, imgDisparity16SRight);

				Ptr<ximgproc::DisparityWLSFilter> wlsFilter = ximgproc::createDisparityWLSFilter(bm);
				wlsFilter->setLambda(lambda / 10.0);
				wlsFilter->setSigmaColor(sigma / 10.0);
				wlsFilter->filter(imgDisparity16SLeft, imgLeft, imgDisparity16S, imgDisparity16SRight);

				Mat imgDisparity8ULeft = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);
				minMaxLoc(imgDisparity16S, &minVal, &maxVal);
				imgDisparity16SLeft.convertTo(imgDisparity8ULeft, CV_8UC1, 255 / (maxVal - minVal));
				resize(imgDisparity8ULeft, imgTotal(rect3), Size(DISP_IMG_WIDTH, DISP_IMG_HEIGHT));
			}
			else {
				bm->compute(imgLeft, imgRight, imgDisparity16S);
			}
		}

		else if (stereoType == VISION_SGBM) {

			Ptr<StereoSGBM> sgbm = StereoSGBM::create(ndisparities, SADWindowSize, 5);

			//-- 3. Calculate the disparity image
			sgbm->compute(imgLeft, imgRight, imgDisparity16S);
		}

		//-- Check its extreme values
		minMaxLoc(imgDisparity16S, &minVal, &maxVal);

		//printf("Min disp: %f Max value: %f \n", minVal, maxVal);

		//-- 4. Display it as a CV_8UC1 image

		//fixDisparity(Mat_<float>(imgDisparity32F), ndisparities, an);
		imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal));
		imgDisparity16S.convertTo(imgDisparity32F, CV_32F);

		resize(imgDisparity8U, imgTotal(rect4), Size(DISP_IMG_WIDTH, DISP_IMG_HEIGHT));

		imshow("Img", imgTotal);

		if(OPENGL_DISP)
			BM23D(imgDisparity32F, Q, (Mat_<Vec3f>) texture);

		cv::waitKey(30);

	}
	return 1;
}

bool StereoVision::runBMStereoPhoto(StereoType stereoType) {
	Mat imgLeft = imread("CalibrationStaticImages\\1.jpg");
	Mat imgRight = imread("CalibrationStaticImages\\2.jpg");
	Mat_<Vec3f> texture = imgLeft.clone();
	cvtColor(imgLeft, imgLeft, CV_BGR2GRAY);
	cvtColor(imgRight, imgRight, CV_BGR2GRAY);

	Mat map11, map12, map21, map22, Q;
	if (!initBMSV(map11, map12, map21, map22, Q)) return 0;

	Mat imgDisparity32F = Mat(imgLeft.rows, imgLeft.cols, CV_32F);
	Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

	//-- 2. Call the constructor for StereoBM
	int preFilterSize = 10; /*Musr be odd*/
	int preFilterCap = 36;

	int ndisparities = 5;   /**< Range of disparity */
	int SADWindowSize = 4; /**< Size of the block window. Must be odd */

	int textureThreshold = 507;
	int uniquenessRatio = 4;
	int speckleWindowSize = 6;
	int speckleRange = 14;

	initDisp(imgLeft.rows, imgLeft.cols);

	cvNamedWindow("Parameters Adjustment");
	cvMoveWindow("Parameters Adjustment", 10, 5);
	cvResizeWindow("Parameters Adjustment", 450, 400);
	createTrackbar("preFilterSize", "Parameters Adjustment", &preFilterSize, 10);
	//createTrackbar("blockSize", "Parameters Adjustment", &blockSize, 40);
	createTrackbar("preFilterCap", "Parameters Adjustment", &preFilterCap, 61);

	createTrackbar("ndisparities", "Parameters Adjustment", &ndisparities, 7);
	createTrackbar("SADWindowSize", "Parameters Adjustment", &SADWindowSize, 12);

	createTrackbar("textureThreshold", "Parameters Adjustment", &textureThreshold, 507);
	createTrackbar("uniquenessRatio", "Parameters Adjustment", &uniquenessRatio, 10);
	createTrackbar("speckleWindowSize", "Parameters Adjustment", &speckleWindowSize, 150);
	createTrackbar("speckleRange", "Parameters Adjustment", &speckleRange, 50);
	while (1) {
		if (stereoType = VISION_BM) {

			Ptr<StereoBM> bm = StereoBM::create(ndisparities * 16, SADWindowSize * 2 + 1);
			//bm->setPreFilterType(CV_STEREO_BM_NORMALIZED_RESPONSE);
			//bm->setPreFilterSize(9);
			bm->setPreFilterCap(preFilterCap + 1);
			bm->setPreFilterSize(preFilterSize * 2 + 5);
			//bm->setBlockSize(blockSize);
			bm->setMinDisparity(-16);
			bm->setTextureThreshold(textureThreshold);
			bm->setUniquenessRatio(uniquenessRatio);
			bm->setSpeckleWindowSize(speckleWindowSize);
			bm->setSpeckleRange(speckleRange);
			bm->setDisp12MaxDiff(1);

			//-- 3. Calculate the disparity image

			bm->compute(imgLeft, imgRight, imgDisparity32F);
		}

		else if (stereoType == VISION_SGBM) {
			ndisparities = 16 * 3;   /**< Range of disparity */
			SADWindowSize = 11; /**< Size of the block window. Must be odd */

			Ptr<StereoSGBM> sgbm = StereoSGBM::create(ndisparities, SADWindowSize, 5);

			//-- 3. Calculate the disparity image
			sgbm->compute(imgLeft, imgRight, imgDisparity32F);
		}

		//-- Check its extreme values
		double minVal; double maxVal;
		minMaxLoc(imgDisparity32F, &minVal, &maxVal);

		//printf("Min disp: %f Max value: %f \n", minVal, maxVal);

		//-- 4. Display it as a CV_8UC1 image
		imgDisparity32F.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal));
		cv::normalize(imgDisparity8U, imgDisparity8U, 0, 255, CV_MINMAX, CV_8UC1);

		namedWindow("Disparity", WINDOW_NORMAL);
		cv::imshow("Disparity", imgDisparity8U);

		BM23D(imgDisparity32F, Q, texture);

		cv::waitKey(100);
	}

	return 1;
}

void StereoVision::fixDisparity(Mat_<float> & disp, int numberOfDisparities, int an)

{
	Mat_<float> disp1;
	float lastPixel = 10;
	float minDisparity = -16;// algorithm parameters that can be modified
	for (int i = 0; i < disp.rows; i++)
	{
		for (int j = numberOfDisparities; j < disp.cols; j++)
		{
			if (disp(i, j) <= minDisparity) disp(i, j) = lastPixel;
			else lastPixel = disp(i, j);
		}
	}

	copyMakeBorder(disp, disp1, an, an, an, an, BORDER_REPLICATE);
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(an * 2 + 1, an * 2 + 1));
	morphologyEx(disp1, disp1, CV_MOP_OPEN, element);
	morphologyEx(disp1, disp1, CV_MOP_CLOSE, element);
	disp = disp1(Range(an, disp.rows - an), Range(an, disp.cols - an)).clone();
}

bool StereoVision::BM23D(Mat& disparity32F, Mat& Q, Mat_<Vec3f> texture) {
	Mat_<Vec3f> XYZ(disparity32F.rows, disparity32F.cols);   // Output point cloud
	//Mat_<float> vec_tmp(4, 1);
	//for (int y = 0; y<disparity32F.rows; ++y) {
	//	for (int x = 0; x<disparity32F.cols; ++x) {
	//		vec_tmp(0) = x;
	//		vec_tmp(1) = y;
	//		vec_tmp(2) = disparity32F.at<float>(y, x);
	//		vec_tmp(3) = 1;

	//		vec_tmp = Q*vec_tmp;
	//		vec_tmp /= vec_tmp(3);

	//		cv::Vec3f &point = XYZ.at<cv::Vec3f>(y, x);
	//		point[0] = vec_tmp(0);
	//		point[1] = vec_tmp(1);
	//		point[2] = vec_tmp(2);
	//	}
	//}

	reprojectImageTo3D(disparity32F, XYZ, Q);

	runStereoDisp(XYZ, texture);


	waitKey(30);

	return 1;
}


/**------------------- Feature Match ----------------------**/

bool StereoVision::initFeatureSV(Mat& K) {
	FileStorage fs0;
	if (!fs0.open("0_result.yml", FileStorage::READ) ||
		!fs0.open("1_result.yml", FileStorage::READ)) {
		cout << "can not find configuration files" << endl;
		return 0;
	}
	fs0["Camera_Matrix"] >> K;
	return 1;
}

bool StereoVision::runFeatureStereoVision(StereoType sType, MatchType mType) {
	vector<vector<KeyPoint>> key_points_for_all;
	vector<Mat> descriptor_for_all;
	vector<vector<Vec3b>> colors_for_all;

	VideoCapture cam1(1);
	VideoCapture cam0(0);
	if (!cam1.isOpened() || !cam0.isOpened()) {
		cout << "can not open camera" << endl;
		return 0;
	}
	cam0.set(CV_CAP_PROP_FRAME_WIDTH, DISP_IMG_WIDTH);
	cam0.set(CV_CAP_PROP_FRAME_HEIGHT, DISP_IMG_HEIGHT);
	cam1.set(CV_CAP_PROP_FRAME_WIDTH, DISP_IMG_WIDTH);
	cam1.set(CV_CAP_PROP_FRAME_HEIGHT, DISP_IMG_HEIGHT);
	//cam0.set(CV_CAP_PROP_BUFFERSIZE, 1);
	//cam1.set(CV_CAP_PROP_BUFFERSIZE, 1);

	Mat K;
	if (!initFeatureSV(K)) return 0;

	Mat imgLeft, imgRight, outImg;
	vector<KeyPoint> key_points_l, key_points_r;
	Mat descriptor_l, descriptor_r;
	Ptr<BFMatcher> matcher;
	Ptr<Feature2D> feature_l, feature_r;
	if (sType == VISION_SURF) {
		matcher = BFMatcher::create(NORM_L2);
		feature_l = xfeatures2d::SURF::create();
		feature_r = xfeatures2d::SURF::create();
		cout << "surf!" << endl;
	}
	else if(sType == VISION_ORB){
		matcher = BFMatcher::create(NORM_HAMMING);
		feature_l = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
		feature_r = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
		cout << "ORB!" << endl;
	}
	else {
		matcher = BFMatcher::create(NORM_L2);
		feature_l = xfeatures2d::SIFT::create();
		feature_r = xfeatures2d::SIFT::create();
		cout << "SIFT!" << endl;
	}

	while (1) {
		key_points_l.clear();
		key_points_r.clear();
		cam0 >> imgLeft;
		cam1 >> imgRight;
		cam0 >> imgLeft;
		cam1 >> imgRight;
		if (imgLeft.empty() || imgRight.empty()) {
			cv::waitKey(30);
			continue;
		}
		flip(imgLeft, imgLeft, 0);
		flip(imgRight, imgRight, 0);
		cvtColor(imgLeft, imgLeft, CV_BGR2GRAY);
		cvtColor(imgRight, imgRight, CV_BGR2GRAY);
		

		GaussianBlur(imgLeft, imgLeft, Size(1,1), 0, 0);
		GaussianBlur(imgRight, imgRight, Size(1, 1), 0, 0);
		equalizeHist(imgLeft, imgLeft);
		equalizeHist(imgRight, imgRight);
		equalizeHist(imgLeft, imgLeft);
		equalizeHist(imgRight, imgRight);

		feature_l->detectAndCompute(imgLeft, noArray(), key_points_l, descriptor_l);
		feature_r->detectAndCompute(imgRight, noArray(), key_points_r, descriptor_r);
		if ((descriptor_l.empty()) || (descriptor_r.empty()) ||
			key_points_l.empty() || key_points_r.empty()) {
			waitKey(30);
			continue;
		}

		//---Calculate Match
		vector<DMatch> matches1, matches2, matches;
		if (mType == KNN_MATCH) {

			vector<vector<DMatch>> knn_matches1, knn_matches2;
			//Match Features
			matcher->knnMatch(descriptor_l, descriptor_r, knn_matches1, 2);
			matcher->knnMatch(descriptor_r, descriptor_l, knn_matches2, 2);

			//获取满足Ratio Test的最小匹配的距离
			matches1 = ratioTest(knn_matches1, 0.7, 5);
			matches2 = ratioTest(knn_matches2, 0.7, 5);

			symmetryTest(matches1, matches2, matches);
		}
		else {
			//Brute Match
			matches.clear();
			matcher->match(descriptor_l, descriptor_r, matches);

			//Best 100
			if (matches.size() > 100) {
				nth_element(matches.begin(), matches.begin() + 99, matches.end());
				matches.erase(matches.begin() + 99, matches.end());
			}
		}

		//---Push back in key points
		vector<Point2f> p1, p2;
		for (std::vector<cv::DMatch>::const_iterator it = matches.cbegin();
			it != matches.cend(); ++it)
		{
			//left keypoints
			p1.push_back(key_points_l[it->queryIdx].pt);
			//right keypoints
			p2.push_back(key_points_r[it->trainIdx].pt);
		}

		//---Draw Matches
		cv::drawMatches(imgLeft, key_points_l, imgRight, key_points_r, matches, outImg,
			cv::Scalar_<double>::all(-1), cv::Scalar_<double>::all(-1), std::vector<char>(0), 2);
		cv::imshow("outImg", outImg);
		cv::waitKey(30);

		//---Calculate R and T
		Mat R, T, structure;
		if (!calRnT(K, p1, p2, R, T)) {
			cout << "calculate R and T failed..." << endl;
			continue;
		}
		//---Re-construction
		if (!keyPoint23D(K, R, T, p1, p2, structure)) {
			cout << "re-construction failed..." << endl;
			continue;
		}
		cout << "re-construction succeed" << endl;
		cout << "R: " << R << endl;
		cout << "T: " << T << endl;
	}

	return 1;
}

bool StereoVision::runFeatureStereoPhoto(StereoType sType, MatchType mType) {
	vector<vector<KeyPoint>> key_points_for_all;
	vector<Mat> descriptor_for_all;
	vector<vector<Vec3b>> colors_for_all;

	Mat imgLeft = imread("CalibrationStaticImages\\1.jpg");
	Mat imgRight = imread("CalibrationStaticImages\\2.jpg");
	cvtColor(imgLeft, imgLeft, CV_BGR2GRAY);
	cvtColor(imgRight, imgRight, CV_BGR2GRAY);

	//读取图像，获取图像特征点，并保存
	Ptr<Feature2D> feature_l = xfeatures2d::SURF::create();
	Ptr<Feature2D> feature_r = xfeatures2d::SURF::create();

	if (imgLeft.empty() || imgRight.empty()) return 0;

	vector<KeyPoint> key_points_l, key_points_r;
	Mat descriptor_l, descriptor_r;
	feature_l->detectAndCompute(imgLeft, noArray(), key_points_l, descriptor_l);
	feature_r->detectAndCompute(imgRight, noArray(), key_points_r, descriptor_r);


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

	Mat outImg;
	vector<char> matchesMask(matches.size(), 0);

	cv::drawMatches(imgLeft, key_points_l, imgRight, key_points_r, matches, outImg,
		Scalar::all(-1), Scalar::all(-1), matchesMask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::imshow("outImg", outImg);



	cv::waitKey(100);
	return 1;
}

vector<DMatch> StereoVision::ratioTest(vector<vector<DMatch>>& rawMatches, 
	double ratioDist, int ratioMinDist) {
	vector<DMatch> matches;
	float min_dist = FLT_MAX;
	for (int r = 0; r < rawMatches.size(); ++r)
	{
		//Ratio Test
		if (rawMatches[r][0].distance > ratioDist * rawMatches[r][1].distance)
			continue;

		float dist = rawMatches[r][0].distance;
		if (dist < min_dist) min_dist = dist;
	}

	for (size_t r = 0; r < rawMatches.size(); ++r)
	{
		//排除不满足Ratio Test的点和匹配距离过大的点
		if (rawMatches[r][0].distance > ratioDist * rawMatches[r][1].distance ||
			rawMatches[r][0].distance > ratioMinDist * max(min_dist, 10.0f))
			continue;

		//保存匹配点
		matches.push_back(rawMatches[r][0]);
	}
	return matches;
}

void StereoVision::symmetryTest(const std::vector<cv::DMatch> &matches1, 
	const std::vector<cv::DMatch> &matches2, std::vector<cv::DMatch>& symMatches) {
	symMatches.clear();
	for (vector<DMatch>::const_iterator matchIt1 = matches1.cbegin(); matchIt1 != matches1.cend(); ++matchIt1)
	{
		for (vector<DMatch>::const_iterator matchIt2 = matches2.cbegin(); matchIt2 != matches2.cend(); ++matchIt2)
		{
			if ((*matchIt1).queryIdx == (*matchIt2).trainIdx && (*matchIt2).queryIdx == (*matchIt1).trainIdx)
			{
				symMatches.push_back(DMatch((*matchIt1).queryIdx, (*matchIt1).trainIdx, (*matchIt1).distance));
				break;
			}
		}
	}
}

bool StereoVision::calRnT(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T) {
	//根据内参矩阵获取相机的焦距和光心坐标（主点坐标）
	double focal_length = 0.5*(K.at<double>(0) + K.at<double>(4));
	Point2d principle_point(K.at<double>(2), K.at<double>(5));

	//根据匹配点求取本征矩阵，使用RANSAC，进一步排除失配点
	Mat mask;
	Mat E = findEssentialMat(Mat(p1), Mat(p2), focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
	if (E.empty()) {
		cout << "can not find Essential Matrix" << endl;
		return false;
	}

	double feasible_count = countNonZero(mask);
	cout << (int)feasible_count << " -in- " << p1.size()  << " ratio:" << feasible_count / p1.size() << endl;
	//对于RANSAC而言，outlier数量大于50%时，结果是不可靠的
	if (feasible_count <= 7 || (feasible_count / p1.size()) < 0.6) {
		cout << "result not reliable" << endl;
		return false;
	}

	//分解本征矩阵，获取相对变换
	int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, noArray());

	//同时位于两个相机前方的点的数量要足够大
	if (((double)pass_count) / feasible_count < 0.7) {
		cout << "not enough common points" << endl;
		return false;
	}

	return true;
}

bool StereoVision::keyPoint23D(Mat& K, Mat& R, Mat& T, vector<Point2f>& p1,
	vector<Point2f>& p2, Mat& structure) {

	Mat proj1(3, 4, CV_32FC1);
	Mat proj2(3, 4, CV_32FC1);

	proj1(Range(0, 3), Range(0, 3)) = Mat::eye(3, 3, CV_32FC1);
	proj1.col(3) = Mat::zeros(3, 1, CV_32FC1);

	R.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
	T.convertTo(proj2.col(3), CV_32FC1);

	Mat fK;
	K.convertTo(fK, CV_32FC1);
	proj1 = fK*proj1;
	proj2 = fK*proj2;

	//三角化重建
	triangulatePoints(proj1, proj2, p1, p2, structure);
	return 1;
}