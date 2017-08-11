#include "StereoDisp.h"

Mat_<Vec3f> imgdata;    // 存放三维坐标数据  
Mat_<Vec3f> texture; // 存放纹理数据  
int height, width, rx = 0, ry = 0;
int eyex = -5, eyey = 4, eyez = -5, atx = 0,aty = 0, atz = 8;
float scalar = 0.01;        //scalar of converting pixel color to float coordinates 

// 功能键（方向键）响应函数 
void special(int key, int x, int y)
{
	switch (key)
	{
	case GLUT_KEY_LEFT:
		atx -= 1;
		glutPostRedisplay();
		break;
	case GLUT_KEY_RIGHT:
		atx += 1;
		glutPostRedisplay();
		break;
	case GLUT_KEY_UP:
		atz += 1;
		glutPostRedisplay();
		break;
	case GLUT_KEY_DOWN:
		atz -= 1;
		glutPostRedisplay();
		break;
	}
}

void keyboard_down(uchar key, int x, int y)
{
	switch (key)
	{
	case 'd':
		eyex += 1;
		glutPostRedisplay();
		break;
	case 'a':
		eyex -= 1;
		glutPostRedisplay();
		break;
	case 'q':
		eyey -= 1;
		glutPostRedisplay();
		break;
	case 'e':
		eyey += 1;
		glutPostRedisplay();
		break;
	case 'w':
		eyez -= 1;
		glutPostRedisplay();
		break;
	case 's':
		eyez += 1;
		glutPostRedisplay();
		break;
	case 'h':
		eyex += 5;
		glutPostRedisplay();
		break;
	case 'f':
		eyex -= 5;
		glutPostRedisplay();
		break;
	case 'r':
		eyey -= 5;
		glutPostRedisplay();
		break;
	case 'y':
		eyey += 5;
		glutPostRedisplay();
		break;
	case 't':
		eyez -= 5;
		glutPostRedisplay();
		break;
	case 'g':
		eyez += 5;
		glutPostRedisplay();
		break;
	case 'l':
		eyex += 10;
		glutPostRedisplay();
		break;
	case 'j':
		eyex -= 10;
		glutPostRedisplay();
		break;
	case 'u':
		eyey -= 10;
		glutPostRedisplay();
		break;
	case 'o':
		eyey += 10;
		glutPostRedisplay();
		break;
	case 'i':
		eyez -= 10;
		glutPostRedisplay();
		break;
	case 'k':
		eyez += 10;
		glutPostRedisplay();
		break;
	}
}


//////////////////////////////////////////////////////////////////////////  
// 三维图像显示响应函数  
void renderScene(void) {

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear color and depth buffers
	glMatrixMode(GL_MODELVIEW);     // To operate on model-view matrix
	glLoadIdentity();// Reset the coordinate system before modifying   
	gluLookAt(eyex, eyey, eyez, eyex + atx, eyey + aty, eyez + atz, 0.0, 1.0, 0.0);    // 根据滑动块位置变换OpenGL摄像机视角  
	std::cout << "eye: " << eyex << ", " << eyey << ", " << eyez << ",\t at: " << eyex + atx << ", " << eyey + aty << ", " << eyez + atz << std::endl;
	float x, y, z;


	//glColor3f(0.5f, 0.5f, 1.0f);							//一次性将所有的颜色设置为一样的
	//glBegin(GL_QUADS);									//  绘制正方形
	//glVertex3f(-100.0f, 100.0f, 0.0f);					// 左上
	//glVertex3f(100.0f, 100.0f, 0.0f);						// 右上
	//glVertex3f(100.0f, -100.0f, 0.0f);					// 左下
	//glVertex3f(-100.0f, -100.0f, 0.0f);					// 右下
	//glEnd();


	glPointSize(1.0);

	glBegin(GL_POINTS);//GL_POINTS  
	for (int i = 0; i<height; i++) {
		for (int j = 0; j<width; j++) {
			glColor3f(texture[i][j][0] / 255, texture[i][j][1] / 255, texture[i][j][2] / 255);    // 将图像纹理赋值到点云上  
			x = imgdata[i][j][0] / scalar;        // 添加负号以获得正确的左右上下方位  
			y = imgdata[i][j][1] / scalar;
			z = imgdata[i][j][2] / scalar;
			glVertex3f(x, y, z);
		}
	}
	glEnd();

	glutSwapBuffers();  // Swap the front and back frame buffers (double buffering)
}

//////////////////////////////////////////////////////////////////////////  
// 窗口变化图像重构响应函数  
void reshape(int w, int h) {
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60, (GLfloat)w / (GLfloat)h, 1.0, 500.0);    // 显示 1 - 500 距离单位（这里是 cm）内的点云  
	glMatrixMode(GL_MODELVIEW);
}

//////////////////////////////////////////////////////////////////////////  
// 载入三维坐标数据  
void load3dDataToGL(const Mat_<Vec3f>& img3d) {
	CvScalar s;
	//accessing the image pixels  
	for (int i = 0; i< height; i++) {
		for (int j = 0; j< width; j++) {        // s.val[0] = x, s.val[1] = y, s.val[2] = z  
			Vec3f v= img3d.at<Vec3f>(i, j);
			imgdata[i][j][0] = v[0];
			imgdata[i][j][1] = -v[1];
			imgdata[i][j][2] = fabs(v[2]);
			//imgdata[i][j][2] = v[2];
		}
	}
}

//////////////////////////////////////////////////////////////////////////  
// 载入左视图纹理数据  
void loadTextureToGL(const Mat& img) {
	//int ind=0;  
	CvScalar ss;
	//accessing the image pixels  
	for (int i = 0; i<height; i++) {
		for (int j = 0; j<width; j++) {
			//OpenCV 是默认 BGR 格式存储彩色图像  
			// ss.val[0] = blue, ss.val[1] = green, ss.val[2] = red  
			texture[i][j][2] = img.at<Vec3f>(i, j)[0];    // OpenGL 则是 RGB 格式存储  
			texture[i][j][1] = img.at<Vec3f>(i, j)[1];
			texture[i][j][0] = img.at<Vec3f>(i, j)[2];
		}
	}
}

void initDisp(int iheight, int iwidth) {
	width = iwidth, height = iheight,
	imgdata.create(height, width);
	texture.create(height, width);
	int argc = 1;
	char *argv[1] = { (char*)"Something" };
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_SINGLE | GLUT_RGBA);
	glutInitWindowPosition(10, 390);
	glutInitWindowSize(450, 390);
	glutCreateWindow("3D disparity image");
}

void runStereoDisp(const Mat_<Vec3f>& img3d, const Mat_<Vec3f>& texture) {

	load3dDataToGL(img3d);            // 载入三维坐标数据  
	loadTextureToGL(texture);        // 载入纹理数据  
	glutReshapeFunc(reshape);            // 窗口变化时重构图像  
	glutDisplayFunc(renderScene);        // 显示三维图像  
	glutSpecialFunc(special);                // 响应方向键按键消息  
	glutKeyboardFunc(keyboard_down);
	glutPostRedisplay();                        // 刷新画面（不用此语句则不能动态更新图像）  
	glutMainLoopEvent();
	std::cout << "-------------disp one frame over-------------" << std::endl;
}