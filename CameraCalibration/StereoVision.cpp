#include "StereoVision.h"

using namespace std;
using namespace cv;

#define IMAGE_HEIGHT 777
#define IMAGE_WIDTH 1000
#define DISP_IMG_HEIGHT 350
#define DISP_IMG_WIDTH 400

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
	stereoRectify(Camera_Matrix0, distCoeff0, Camera_Matrix1, distCoeff1, imgSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY);

	initUndistortRectifyMap(Camera_Matrix0, distCoeff0, R1, P1, imgSize, CV_16SC2, map11, map12);
	initUndistortRectifyMap(Camera_Matrix1, distCoeff1, R2, P2, imgSize, CV_16SC2, map21, map22);

	return 1;
}

bool StereoVision::runBMStereoVision(StereoType stereoType) {

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

	Mat map11, map12, map21, map22, Q;
	if (!initBMSV(map11, map12, map21, map22, Q)) return 0;

	Rect rect1, rect2;
	rect1 = Rect(0, 0, DISP_IMG_WIDTH, DISP_IMG_HEIGHT);
	rect2 = Rect(DISP_IMG_WIDTH, 0, DISP_IMG_WIDTH, DISP_IMG_HEIGHT);

	Mat imgLeft, imgRight, imgLeft_tmp, imgRight_tmp, imgTotal;
	while (1)
	{
		cam0 >> imgLeft;
		cam1 >> imgRight;
		flip(imgLeft, imgLeft, 0);
		flip(imgRight, imgRight, 0);
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
		cv::imshow("Img", imgTotal);

		Mat imgDisparity32F = Mat(imgLeft.rows, imgLeft.cols, CV_32F);
		Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

		if (imgLeft.empty() || imgRight.empty())
		{
			std::cout << " --(!) Error reading images " << std::endl;
			cv::waitKey(1000);
			continue;
		}

		//-- 2. Call the constructor for StereoBM
		int ndisparities, SADWindowSize;

		if (stereoType = VISION_BM) {
			ndisparities = 16 * 3;   /**< Range of disparity */
			SADWindowSize = 19; /**< Size of the block window. Must be odd */

			Ptr<StereoBM> bm = StereoBM::create(ndisparities, SADWindowSize);
			//bm->setPreFilterType(CV_STEREO_BM_NORMALIZED_RESPONSE);
			//bm->setPreFilterSize(9);
			bm->setPreFilterCap(31);
			bm->setBlockSize(21);
			bm->setMinDisparity(-16);
			bm->setNumDisparities(80);
			//bm->setTextureThreshold(10);
			bm->setUniquenessRatio(5);
			bm->setSpeckleWindowSize(100);
			bm->setSpeckleRange(32);

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
		namedWindow("Disparity before fix", WINDOW_NORMAL);
		imgDisparity32F.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal));
		Mat imgDisparity_tmp = imgDisparity8U.clone();
		cv::imshow("Disparity before fix", imgDisparity_tmp);
		cv::waitKey(30);

		fixDisparity(Mat_<float>(imgDisparity32F), ndisparities);
		imgDisparity32F.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal));

		namedWindow("Disparity", WINDOW_NORMAL);
		cv::imshow("Disparity", imgDisparity8U);


		cv::waitKey(30);

	}
	return 1;
}

bool StereoVision::runBMStereoPhoto(StereoType stereoType) {
	Mat imgLeft = imread("CalibrationStaticImages\\1.jpg");
	Mat imgRight = imread("CalibrationStaticImages\\2.jpg");
	cvtColor(imgLeft, imgLeft, CV_BGR2GRAY);
	cvtColor(imgRight, imgRight, CV_BGR2GRAY);

	Mat imgDisparity32F = Mat(imgLeft.rows, imgLeft.cols, CV_32F);
	Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

	//-- 2. Call the constructor for StereoBM
	int ndisparities, SADWindowSize;

	if (stereoType = VISION_BM) {
		ndisparities = 16 * 3;   /**< Range of disparity */
		SADWindowSize = 19; /**< Size of the block window. Must be odd */

		Ptr<StereoBM> bm = StereoBM::create(ndisparities, SADWindowSize);
		//bm->setPreFilterType(CV_STEREO_BM_NORMALIZED_RESPONSE);
		//bm->setPreFilterSize(9);
		bm->setPreFilterCap(31);
		bm->setBlockSize(21);
		bm->setMinDisparity(-16);
		bm->setNumDisparities(80);
		//bm->setTextureThreshold(10);
		bm->setUniquenessRatio(5);
		bm->setSpeckleWindowSize(100);
		bm->setSpeckleRange(32);

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
	cv::waitKey(100);

	return 1;
}

void StereoVision::fixDisparity(Mat_<float> & disp, int numberOfDisparities)

{
	Mat_<float> disp1;
	float lastPixel = 10;
	float minDisparity = 23;// algorithm parameters that can be modified
	for (int i = 0; i < disp.rows; i++)
	{
		for (int j = numberOfDisparities; j < disp.cols; j++)
		{
			if (disp(i, j) <= minDisparity) disp(i, j) = lastPixel;
			else lastPixel = disp(i, j);
		}
	}
	int an = 4;	// algorithm parameters that can be modified
	copyMakeBorder(disp, disp1, an, an, an, an, BORDER_REPLICATE);
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(an * 2 + 1, an * 2 + 1));
	morphologyEx(disp1, disp1, CV_MOP_OPEN, element);
	morphologyEx(disp1, disp1, CV_MOP_CLOSE, element);
	disp = disp1(Range(an, disp.rows - an), Range(an, disp.cols - an)).clone();
}

void StereoVision::BM23D(Mat& disparity32F, Mat& Q) {
	Mat_<Vec3f> XYZ(disparity32F.rows, disparity32F.cols);   // Output point cloud
	Mat_<float> vec_tmp(4, 1);
	for (int y = 0; y<disparity32F.rows; ++y) {
		for (int x = 0; x<disparity32F.cols; ++x) {
			vec_tmp(0) = x;
			vec_tmp(1) = y;
			vec_tmp(2) = disparity32F.at<float>(y, x);
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
	else {
		matcher = BFMatcher::create(NORM_HAMMING);
		feature_l = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
		feature_r = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
		cout << "ORB!" << endl;
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
	if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6) {
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