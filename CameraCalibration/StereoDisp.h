#pragma once

#include <iostream>
#include <freeglut.h>
#include <opencv2/core/core.hpp>

using namespace cv;

void initDisp(int iheight, int iwidth);
void runStereoDisp(const Mat_<Vec3f>& img3d, const Mat_<Vec3f>& texture);