#include "TakePictures.h"

#define IMAGE_HEIGHT 768
#define IMAGE_WIDTH 1024
#define DISP_IMG_HEIGHT 400
#define DISP_IMG_WIDTH 500

using namespace std;
using namespace cv;

TakePictures::TakePictures()
{
}

TakePictures::~TakePictures()
{
}

void TakePictures::runTakePictures() {
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
	int cnt = 0;
	int c;
	string fileName;
	cout << "Press enter to skip the current frame, any key else to save the picture" << endl;
	while (1) {
		cam1 >> imgLeft;
		cam0 >> imgRight;
		flip(imgLeft, imgLeft, 0);
		flip(imgRight, imgRight, 0);
		imgTotal.create(Size(DISP_IMG_WIDTH * 2, DISP_IMG_HEIGHT), CV_8UC3);
		resize(imgLeft, imgTotal(rect1), Size(DISP_IMG_WIDTH, DISP_IMG_HEIGHT));
		resize(imgRight, imgTotal(rect2), Size(DISP_IMG_WIDTH, DISP_IMG_HEIGHT));
		imshow("img", imgTotal);
		waitKey(30);
		fileName = "image" + to_string(cnt);

		c = getchar();

		if (c == '\n') {
			cout << "ignore" << endl;
			continue;
		}
		if (c == 'q') {
			cout << cnt << " pictures saved" << endl;
			return;
		}
		cout << "writing images : " << cnt << endl;
		imwrite("CalibrationImages\\" + fileName + "_0.jpg", imgLeft);
		imwrite("CalibrationImages\\" + fileName + "_1.jpg", imgRight);
		++cnt;
	}
}