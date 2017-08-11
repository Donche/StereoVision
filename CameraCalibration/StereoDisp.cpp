#include "StereoDisp.h"

Mat_<Vec3f> imgdata;    // �����ά��������  
Mat_<Vec3f> texture; // �����������  
int height, width, rx = 0, ry = 0;
int eyex = -5, eyey = 4, eyez = -5, atx = 0,aty = 0, atz = 8;
float scalar = 0.01;        //scalar of converting pixel color to float coordinates 

// ���ܼ������������Ӧ���� 
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
// ��άͼ����ʾ��Ӧ����  
void renderScene(void) {

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear color and depth buffers
	glMatrixMode(GL_MODELVIEW);     // To operate on model-view matrix
	glLoadIdentity();// Reset the coordinate system before modifying   
	gluLookAt(eyex, eyey, eyez, eyex + atx, eyey + aty, eyez + atz, 0.0, 1.0, 0.0);    // ���ݻ�����λ�ñ任OpenGL������ӽ�  
	std::cout << "eye: " << eyex << ", " << eyey << ", " << eyez << ",\t at: " << eyex + atx << ", " << eyey + aty << ", " << eyez + atz << std::endl;
	float x, y, z;


	//glColor3f(0.5f, 0.5f, 1.0f);							//һ���Խ����е���ɫ����Ϊһ����
	//glBegin(GL_QUADS);									//  ����������
	//glVertex3f(-100.0f, 100.0f, 0.0f);					// ����
	//glVertex3f(100.0f, 100.0f, 0.0f);						// ����
	//glVertex3f(100.0f, -100.0f, 0.0f);					// ����
	//glVertex3f(-100.0f, -100.0f, 0.0f);					// ����
	//glEnd();


	glPointSize(1.0);

	glBegin(GL_POINTS);//GL_POINTS  
	for (int i = 0; i<height; i++) {
		for (int j = 0; j<width; j++) {
			glColor3f(texture[i][j][0] / 255, texture[i][j][1] / 255, texture[i][j][2] / 255);    // ��ͼ������ֵ��������  
			x = imgdata[i][j][0] / scalar;        // ��Ӹ����Ի����ȷ���������·�λ  
			y = imgdata[i][j][1] / scalar;
			z = imgdata[i][j][2] / scalar;
			glVertex3f(x, y, z);
		}
	}
	glEnd();

	glutSwapBuffers();  // Swap the front and back frame buffers (double buffering)
}

//////////////////////////////////////////////////////////////////////////  
// ���ڱ仯ͼ���ع���Ӧ����  
void reshape(int w, int h) {
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60, (GLfloat)w / (GLfloat)h, 1.0, 500.0);    // ��ʾ 1 - 500 ���뵥λ�������� cm���ڵĵ���  
	glMatrixMode(GL_MODELVIEW);
}

//////////////////////////////////////////////////////////////////////////  
// ������ά��������  
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
// ��������ͼ��������  
void loadTextureToGL(const Mat& img) {
	//int ind=0;  
	CvScalar ss;
	//accessing the image pixels  
	for (int i = 0; i<height; i++) {
		for (int j = 0; j<width; j++) {
			//OpenCV ��Ĭ�� BGR ��ʽ�洢��ɫͼ��  
			// ss.val[0] = blue, ss.val[1] = green, ss.val[2] = red  
			texture[i][j][2] = img.at<Vec3f>(i, j)[0];    // OpenGL ���� RGB ��ʽ�洢  
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

	load3dDataToGL(img3d);            // ������ά��������  
	loadTextureToGL(texture);        // ������������  
	glutReshapeFunc(reshape);            // ���ڱ仯ʱ�ع�ͼ��  
	glutDisplayFunc(renderScene);        // ��ʾ��άͼ��  
	glutSpecialFunc(special);                // ��Ӧ�����������Ϣ  
	glutKeyboardFunc(keyboard_down);
	glutPostRedisplay();                        // ˢ�»��棨���ô�������ܶ�̬����ͼ��  
	glutMainLoopEvent();
	std::cout << "-------------disp one frame over-------------" << std::endl;
}