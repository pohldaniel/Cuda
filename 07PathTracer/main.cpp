#define NOMINMAX

#include <windows.h>
#include <stdio.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#include "pathtracer.h"

#include "glm-0.9.2.6/glm/glm.hpp"
#include "glm-0.9.2.6/glm/gtc/matrix_transform.hpp"
#include "glm-0.9.2.6/glm/gtc/type_ptr.hpp"

#include "Extension.h"
#include "Quad.h"

#include "glhelper.h"
#include "math.h"
#include "matrix.h"
#include "obj.h"

#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))
#define CLAMP(x, y, z) ((x)<(y)?(y):((x)>(z)?(z):(x)))

#define WIDTH	1280
#define HEIGHT	720

Quad *quad;
GLuint m_texture;
float *fb;
struct cudaGraphicsResource *cuda_tex_result_resource;
int num_pixels = WIDTH * HEIGHT;
size_t fb_size = 4 * num_pixels * sizeof(float);
cudaArray *texture_ptr;

POINT g_OldCursorPos;
bool g_enableVerticalSync;

LRESULT CALLBACK winProc(HWND hWnd, UINT message, WPARAM wParma, LPARAM lParam);
void initApp(HWND hWnd);
void enableVerticalSync(bool enableVerticalSync);
void generateCUDAImage();


// abstract base class: the plane, sphere, and triangle objects are derived from this
int BPP = 32;
int MAX_BOUNCES = 5;
double focal_distance = 1800.0; // updated below to center of min and max
double blur_radius = 4;// 32.0;

					   //cTimer t0; double elapsed0;

					   /*timespec current;
					   clock_gettime(CLOCK_REALTIME, &current);
					   srand(current.tv_sec + current.tv_nsec / 1000000000.0);*/

bool active = true;
unsigned int samples = 1;

void savePPM(unsigned char* buffer, int width, int height, int bpp, const char* image_out) {
	std::fstream of(image_out, std::ios_base::out | std::ios_base::trunc);
	of << "P3" << std::endl;
	of << width << " " << height << std::endl;
	of << "255" << std::endl;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			of << (int)buffer[j * width * bpp / 8 + i * bpp / 8 + 0] << " "
				<< (int)buffer[j * width * bpp / 8 + i * bpp / 8 + 1] << " "
				<< (int)buffer[j * width * bpp / 8 + i * bpp / 8 + 2] << " ";
		}
		of << std::endl;
	}
}

class cObject {
public:
	int type;
	int material;
	vector3 color;
	vector3 emissive;

	cObject();
	cObject(vector3 color, vector3 emissive, int material);
	~cObject();
	virtual void applyCamera(cMatrix& camera) = 0;
	virtual void save() = 0;
};
std::vector<cObject*> objects;
cObject::cObject() : color(vector3(1.0, 1.0, 1.0)), emissive(vector3(0.0, 0.0, 0.0)), material(DIFFUSE) { }
cObject::cObject(vector3 color, vector3 emissive, int material) : color(color), emissive(emissive), material(material) { }
cObject::~cObject() { };//std::cout << "object destroyed" << std::endl; }


class cPlane : public cObject {
public:
	vector3 normal, normal_;
	vector3 point, point_;

	cPlane();
	cPlane(vector3 normal, vector3 point, vector3 color, vector3 emissive, int material);
	~cPlane();
	void applyCamera(cMatrix& camera);
	void save();
};

cPlane::cPlane() : normal(vector3(0.0, 0.0, 1.0)), point(vector3()), normal_(vector3(0.0, 0.0, 1.0)), point_(vector3()), cObject(vector3(1.0, 1.0, 1.0), vector3(0.0, 0.0, 0.0), material) { this->type = PLANE; }
cPlane::cPlane(vector3 normal, vector3 point, vector3 color, vector3 emissive, int material) : normal(normal), point(point), normal_(normal), point_(point), cObject(color, emissive, material) { this->type = PLANE; }
cPlane::~cPlane() { std::cout << "plane destroyed" << std::endl; }
void cPlane::applyCamera(cMatrix& camera) {
	vector3 p = this->point_ + this->normal_;
	this->point = camera.mult(this->point_);
	p = camera.mult(p);
	this->normal = (p - this->point).unit();
}
void cPlane::save() {
	this->normal_ = this->normal;
	this->point_ = this->point;
}


class cSphere : public cObject {
public:
	vector3 center, center_;
	double radius, radius_;

	cSphere();
	cSphere(vector3 center, double radius, vector3 color, vector3 emissive, int material);
	~cSphere();
	void applyCamera(cMatrix& camera);
	void save();
};

cSphere::cSphere() : center(vector3(0.0, 0.0, 0.0)), center_(vector3(0.0, 0.0, 0.0)), radius(1.0), radius_(1.0), cObject(vector3(1.0, 1.0, 1.0), vector3(0.0, 0.0, 0.0), material) { this->type = SPHERE; }
cSphere::cSphere(vector3 center, double radius, vector3 color, vector3 emissive, int material) : center(center), center_(center), radius(radius), radius_(radius), cObject(color, emissive, material) { this->type = SPHERE; }
cSphere::~cSphere() { std::cout << "sphere destroyed" << std::endl; }
void cSphere::applyCamera(cMatrix& camera) {
	this->center = camera.mult(this->center_);
}
void cSphere::save() {
	this->center_ = this->center;
	this->radius_ = this->radius;
}

class cTriangle : public cObject {
public:
	vector3 p0, p1, p2, p0_, p1_, p2_;
	vector3 n0, n1, n2, n0_, n1_, n2_;
	vector3 n, n_;

	cTriangle();
	cTriangle(vector3 p0, vector3 p1, vector3 p2, vector3 n0, vector3 n1, vector3 n2, vector3 color, vector3 emissive, int material);
	~cTriangle();
	void applyCamera(cMatrix& camera);
	void save();
};

cTriangle::cTriangle() :
	p0(vector3(0.0, 1.0, 0.0)),
	p1(vector3(-1.0, 0.0, 0.0)),
	p2(vector3(1.0, 0.0, 0.0)),
	p0_(vector3(0.0, 1.0, 0.0)),
	p1_(vector3(-1.0, 0.0, 0.0)),
	p2_(vector3(1.0, 0.0, 0.0)),
	n0(vector3(0.0, 0.0, 1.0)),
	n1(vector3(0.0, 0.0, 1.0)),
	n2(vector3(0.0, 0.0, 1.0)),
	n0_(vector3(0.0, 0.0, 1.0)),
	n1_(vector3(0.0, 0.0, 1.0)),
	n2_(vector3(0.0, 0.0, 1.0)),
	cObject(vector3(1.0, 1.0, 1.0), vector3(0.0, 0.0, 0.0), material) {
	this->type = TRIANGLE;
	this->n = this->n_ = ((p1 - p0).cross(p2 - p0)).unit();
}
cTriangle::cTriangle(vector3 p0, vector3 p1, vector3 p2, vector3 n0, vector3 n1, vector3 n2, vector3 color, vector3 emissive, int material) :
	p0(p0),
	p1(p1),
	p2(p2),
	n0(n0),
	n1(n1),
	n2(n2),
	p0_(p0),
	p1_(p1),
	p2_(p2),
	n0_(n0),
	n1_(n1),
	n2_(n2),
	cObject(color, emissive, material) {
	this->type = TRIANGLE;
	this->n = this->n_ = ((p1 - p0).cross(p2 - p0)).unit();
}
cTriangle::~cTriangle() { std::cout << "plane destroyed" << std::endl; }
void cTriangle::applyCamera(cMatrix& camera) {
	vector3 _p0, _p1, _p2;
	vector3 _pn;

	_pn = this->p0_ + this->n_;
	_p0 = this->p0_ + this->n0_;
	_p1 = this->p1_ + this->n1_;
	_p2 = this->p2_ + this->n2_;

	this->p0 = camera.mult(this->p0_);
	this->p1 = camera.mult(this->p1_);
	this->p2 = camera.mult(this->p2_);

	_pn = camera.mult(_pn);
	_p0 = camera.mult(_p0);
	_p1 = camera.mult(_p1);
	_p2 = camera.mult(_p2);

	this->n = (_pn - this->p0).unit();
	this->n0 = (_p0 - this->p0).unit();
	this->n1 = (_p1 - this->p1).unit();
	this->n2 = (_p2 - this->p2).unit();

}
void cTriangle::save() {
	this->p0_ = this->p0;
	this->p1_ = this->p1;
	this->p2_ = this->p2;
	this->n0_ = this->n0;
	this->n1_ = this->n1;
	this->n2_ = this->n2;
	this->n_ = this->n;
}

struct vertex_ {
	double x, y, z;
	double tx, ty, tz;
	double t[6];
};

unsigned char *current_buffer_host;
unsigned char *current_buffer_device;
double *current_doubles_device;
__object *current_objects_device;
float *rand_device;
_bounding_box b;
int offsetx = 0, offsety = 0;
bool sleep = false;
__object* _objects;

////////////////////////////////////////////////////////////////////////////////
extern "C" bool initializePathTracer(_bounding_box& b, unsigned char*& current_buffer_host, unsigned char *& current_buffer_device, double*& current_doubles_device, __object* _objects, __object*& current_objects_device, int object_count, float*& rand_device, int max_bounces, int width, int height, double minx, double miny, double minz, double maxx, double maxy, double maxz);
extern "C" bool releasePathTracer(_bounding_box& b, unsigned char*& current_buffer_host, unsigned char *& current_buffer_device, double*& current_doubles_device, __object*& current_objects_device, float*& rand_device, int width, int height);
extern "C" bool runPathTracer(_bounding_box& b, float *fb, unsigned char*& current_buffer_host, unsigned char *& current_buffer_device, double*& current_doubles_device, float*& rand_device, unsigned long long samples, __object objects[], __object*& current_objects_device, int object_count, int max_bounces, int width, int height, int offsetx, int offsety, double focal_distance, double blur_radius);


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {

	AllocConsole();
	AttachConsole(GetCurrentProcessId());
	freopen("CON", "w", stdout);
	SetConsoleTitle("Debug console");

	MoveWindow(GetConsoleWindow(), 1300, 0, 550, 300, true);
	
	WNDCLASSEX		windowClass;		// window class
	HWND			hwnd;				// window handle
	MSG				msg;				// message
	HDC				hdc;				// device context handle

										// fill out the window class structure
	windowClass.cbSize = sizeof(WNDCLASSEX);
	windowClass.style = CS_HREDRAW | CS_VREDRAW;
	windowClass.lpfnWndProc = winProc;
	windowClass.cbClsExtra = 0;
	windowClass.cbWndExtra = 0;
	windowClass.hInstance = hInstance;
	windowClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);		// default icon
	windowClass.hCursor = LoadCursor(NULL, IDC_ARROW);			// default arrow
	windowClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);	// white background
	windowClass.lpszMenuName = NULL;									// no menu
	windowClass.lpszClassName = "WINDOWCLASS";
	windowClass.hIconSm = LoadIcon(NULL, IDI_WINLOGO);			// windows logo small icon

																// register the windows class
	if (!RegisterClassEx(&windowClass))
		return 0;

	// class registered, so now create our window
	hwnd = CreateWindowEx(
		NULL,									// extended style
		"WINDOWCLASS",							// class name
		"CUDA Pathtracer",						// app name
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT,			// x,y coordinate
		WIDTH,
		HEIGHT,							// width, height
		NULL,									// handle to parent
		NULL,									// handle to menu
		hInstance,								// application instance
		NULL);									// no extra params

											// check if window creation failed (hwnd would equal NULL)
	if (!hwnd)
		return 0;

	ShowWindow(hwnd, SW_SHOW);			// display the window
	UpdateWindow(hwnd);					// update the window

	initApp(hwnd);

	// main message loop
	while (true) {

		// Did we recieve a message, or are we idling ?
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
			// test if this is a quit
			if (msg.message == WM_QUIT) break;
			// translate and dispatch message
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}else {			
			runPathTracer(b, fb, current_buffer_host, current_buffer_device, current_doubles_device, rand_device, samples, _objects, current_objects_device, objects.size(), MAX_BOUNCES, WIDTH, HEIGHT, offsetx, offsety, focal_distance, blur_radius);
			generateCUDAImage();
					
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			quad->render(m_texture);

			offsetx += 64;
			if (offsetx > (WIDTH - 64)) {
				offsetx = 0;
				offsety += 64;
				if (offsety > (HEIGHT - 64)) {
					offsetx = offsety = 0;
					samples++;
				}
			}
			
			hdc = GetDC(hwnd);
			SwapBuffers(hdc);
			ReleaseDC(hwnd, hdc);
		}
	} // end while

	releasePathTracer(b, current_buffer_host, current_buffer_device, current_doubles_device, current_objects_device, rand_device, WIDTH, HEIGHT);

	// release the objects
	for (int i = 0; i < objects.size(); i++) delete objects[i];

	return msg.wParam;
}

LRESULT CALLBACK winProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {

	static HGLRC hRC;					// rendering context
	static HDC hDC;						// device context
	POINT pt;
	RECT rect;

	switch (message) {

	case WM_DESTROY: {

		PostQuitMessage(0);
		return 0;
	}

	case WM_CREATE: {

		GetClientRect(hWnd, &rect);
		g_OldCursorPos.x = rect.right / 2;
		g_OldCursorPos.y = rect.bottom / 2;
		pt.x = rect.right / 2;
		pt.y = rect.bottom / 2;
		//SetCursorPos(pt.x, pt.y);
		// set the cursor to the middle of the window and capture the window via "SendMessage"
		SendMessage(hWnd, WM_LBUTTONDOWN, MK_LBUTTON, MAKELPARAM(pt.x, pt.y));
		return 0;
	}break;

	case WM_LBUTTONDOWN: { // Capture the mouse

						   //setCursortoMiddle(hWnd);
						   //SetCapture(hWnd);

		return 0;
	} break;

	case WM_KEYDOWN: {

		switch (wParam) {

		case VK_ESCAPE: {

			PostQuitMessage(0);
			return 0;

		}break;
		case VK_SPACE: {

			ReleaseCapture();
			return 0;

		}break;

		case 'v': case 'V': {
			enableVerticalSync(!g_enableVerticalSync);
			return 0;

		}break;


			return 0;
		}break;

		return 0;
	}break;

	case WM_SIZE: {

		int _height = HIWORD(lParam);		// retrieve width and height
		int _width = LOWORD(lParam);

		if (_height == 0) {					// avoid divide by zero
			_height = 1;
		}

		glViewport(0, 0, _width, _height);

		return 0;
	}break;

	default:
		break;
	}
	return (DefWindowProc(hWnd, message, wParam, lParam));
}

void initApp(HWND hWnd) {

	static HGLRC hRC;					// rendering context
	static HDC hDC;						// device context

	hDC = GetDC(hWnd);
	int nPixelFormat;					// our pixel format index

	static PIXELFORMATDESCRIPTOR pfd = {
		sizeof(PIXELFORMATDESCRIPTOR),	// size of structure
		1,								// default version
		PFD_DRAW_TO_WINDOW |			// window drawing support
		PFD_SUPPORT_OPENGL |			// OpenGL support
		PFD_DOUBLEBUFFER,				// double buffering support
		PFD_TYPE_RGBA,					// RGBA color mode
		32,								// 32 bit color mode
		0, 0, 0, 0, 0, 0,				// ignore color bits, non-palettized mode
		0,								// no alpha buffer
		0,								// ignore shift bit
		0,								// no accumulation buffer
		0, 0, 0, 0,						// ignore accumulation bits
		16,								// 16 bit z-buffer size
		0,								// no stencil buffer
		0,								// no auxiliary buffer
		PFD_MAIN_PLANE,					// main drawing plane
		0,								// reserved
		0, 0, 0 };						// layer masks ignored

	nPixelFormat = ChoosePixelFormat(hDC, &pfd);	// choose best matching pixel format
	SetPixelFormat(hDC, nPixelFormat, &pfd);		// set pixel format to device context


	hRC = wglCreateContext(hDC);				// create rendering context and make it current
	wglMakeCurrent(hDC, hRC);
	enableVerticalSync(true);

	glEnable(GL_DEPTH_TEST);					// hidden surface removal

	quad = new Quad(1.0f, 1.0f);

	// create a texture
	glGenTextures(1, &m_texture);
	glBindTexture(GL_TEXTURE_2D, m_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WIDTH, HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	// register this texture with CUDA
	cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, m_texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);

	//Init CUDABuffer
	cudaMallocManaged((void **)&fb, fb_size);

	// two planes (sky and ground)
	objects.push_back(new cPlane(vector3(0, -1, 0), vector3(0, 8192, 0), vector3(0, 0, 0), vector3(1, 1, 1), DIFFUSE)); // top
	objects.push_back(new cPlane(vector3(0, 1, 0), vector3(0, 20, 0), vector3(0.8, 0.8, 0.8), vector3(), DIFFUSE)); // bottom

																													// load the obj model
	cObj obj2("monkey_subdivided.obj");
	for (int i = 0000; i < MIN(10000, obj2.faces.size()); i++) {
		objects.push_back(new cTriangle(vector3(obj2.vertices[obj2.faces[i].vertex[0]].v[0], obj2.vertices[obj2.faces[i].vertex[0]].v[1], obj2.vertices[obj2.faces[i].vertex[0]].v[2]) * 100 + vector3(0, 60, -70),
			vector3(obj2.vertices[obj2.faces[i].vertex[1]].v[0], obj2.vertices[obj2.faces[i].vertex[1]].v[1], obj2.vertices[obj2.faces[i].vertex[1]].v[2]) * 100 + vector3(0, 60, -70),
			vector3(obj2.vertices[obj2.faces[i].vertex[2]].v[0], obj2.vertices[obj2.faces[i].vertex[2]].v[1], obj2.vertices[obj2.faces[i].vertex[2]].v[2]) * 100 + vector3(0, 60, -70),
			vector3(obj2.normals[obj2.faces[i].normal[0]].v[0], obj2.normals[obj2.faces[i].normal[0]].v[1], obj2.normals[obj2.faces[i].normal[0]].v[2]),
			vector3(obj2.normals[obj2.faces[i].normal[1]].v[0], obj2.normals[obj2.faces[i].normal[1]].v[1], obj2.normals[obj2.faces[i].normal[1]].v[2]),
			vector3(obj2.normals[obj2.faces[i].normal[2]].v[0], obj2.normals[obj2.faces[i].normal[2]].v[1], obj2.normals[obj2.faces[i].normal[2]].v[2]),
			vector3(0.7, 1.0, 0.7), vector3(0, 0.0, 0), DIFFUSE)); // top
	}

	// add some spheres resting on the ground
	for (int i = 0; i < 40; i++) {
		double sx = rand() / (double)RAND_MAX;
		double sy = rand() / (double)RAND_MAX;
		double sz = rand() / (double)RAND_MAX;
		double sr = rand() / (double)RAND_MAX;

		sx -= 0.5; sy -= 0.5; sz -= 0.5; sr += 0.5;
		sx *= 400.0*WIDTH / HEIGHT; sy *= 200; sz *= 400; sr *= 15;
		int type = rand() % 2;

		sy = sr + 20; //100;
		double r = type == 0 ? rand() / (double)RAND_MAX : 0.9;
		double g = type == 0 ? rand() / (double)RAND_MAX : 0.9;
		double b = type == 0 ? rand() / (double)RAND_MAX : 0.9;

		objects.push_back(new cSphere(vector3(sx, sy, sz), sr, vector3(r, g, b), vector3(), type)); // light source
	}

	// set up the camera and apply it to the primitives
	{
		vector3  origin(100, 150, 500);
		vector3 forward = (vector3(20, 140, 0) - origin).unit();
		vector3 up(0, 1, 0);
		vector3 right = (forward.cross(up)).unit();
		up = (right.cross(forward)).unit();

		double cam[] = { right.x,   right.y,   right.z, -(right*origin),
			up.x,      up.y,      up.z, -(up*origin),
			forward.x, forward.y, forward.z, -(forward*origin),
			0,         0,         0,         1 };

		cMatrix camera(4, 4, cam);
		//camera.output();

		for (int i = 0; i < objects.size(); i++) {
			objects[i]->applyCamera(camera);
			objects[i]->save();
		}
	}

	double minx = 10000000, miny = 10000000, minz = 10000000;
	double maxx = -10000000, maxy = -10000000, maxz = -10000000;

	
	// copy the transformed objects into an __object array
	_objects = new __object[objects.size()];
	for (int i = 0; i < objects.size(); i++) {

		_objects[i].type = objects[i]->type;
		_objects[i].material = objects[i]->material;

		_objects[i].color.x = objects[i]->color.x;
		_objects[i].color.y = objects[i]->color.y;
		_objects[i].color.z = objects[i]->color.z;

		_objects[i].emission.x = objects[i]->emissive.x;
		_objects[i].emission.y = objects[i]->emissive.y;
		_objects[i].emission.z = objects[i]->emissive.z;

		if (objects[i]->type == PLANE) {
			_objects[i].normal.x = ((cPlane *)objects[i])->normal.x;
			_objects[i].normal.y = ((cPlane *)objects[i])->normal.y;
			_objects[i].normal.z = ((cPlane *)objects[i])->normal.z;
			_objects[i].point.x = ((cPlane *)objects[i])->point.x;
			_objects[i].point.y = ((cPlane *)objects[i])->point.y;
			_objects[i].point.z = ((cPlane *)objects[i])->point.z;
		}else if (objects[i]->type == SPHERE) {
			_objects[i].center.x = ((cSphere *)objects[i])->center.x;
			_objects[i].center.y = ((cSphere *)objects[i])->center.y;
			_objects[i].center.z = ((cSphere *)objects[i])->center.z;
			_objects[i].radius = ((cSphere *)objects[i])->radius;

			minx = MIN(minx, _objects[i].center.x - _objects[i].radius);
			miny = MIN(miny, _objects[i].center.y - _objects[i].radius);
			minz = MIN(minz, _objects[i].center.z - _objects[i].radius);
			maxx = MAX(maxx, _objects[i].center.x + _objects[i].radius);
			maxy = MAX(maxy, _objects[i].center.y + _objects[i].radius);
			maxz = MAX(maxz, _objects[i].center.z + _objects[i].radius);
		}else if (objects[i]->type == TRIANGLE) {
			_objects[i].p0.x = ((cTriangle *)objects[i])->p0.x;
			_objects[i].p0.y = ((cTriangle *)objects[i])->p0.y;
			_objects[i].p0.z = ((cTriangle *)objects[i])->p0.z;
			_objects[i].p1.x = ((cTriangle *)objects[i])->p1.x;
			_objects[i].p1.y = ((cTriangle *)objects[i])->p1.y;
			_objects[i].p1.z = ((cTriangle *)objects[i])->p1.z;
			_objects[i].p2.x = ((cTriangle *)objects[i])->p2.x;
			_objects[i].p2.y = ((cTriangle *)objects[i])->p2.y;
			_objects[i].p2.z = ((cTriangle *)objects[i])->p2.z;
			_objects[i].n0.x = ((cTriangle *)objects[i])->n0.x;
			_objects[i].n0.y = ((cTriangle *)objects[i])->n0.y;
			_objects[i].n0.z = ((cTriangle *)objects[i])->n0.z;
			_objects[i].n1.x = ((cTriangle *)objects[i])->n1.x;
			_objects[i].n1.y = ((cTriangle *)objects[i])->n1.y;
			_objects[i].n1.z = ((cTriangle *)objects[i])->n1.z;
			_objects[i].n2.x = ((cTriangle *)objects[i])->n2.x;
			_objects[i].n2.y = ((cTriangle *)objects[i])->n2.y;
			_objects[i].n2.z = ((cTriangle *)objects[i])->n2.z;
			_objects[i].n.x = ((cTriangle *)objects[i])->n.x;
			_objects[i].n.y = ((cTriangle *)objects[i])->n.y;
			_objects[i].n.z = ((cTriangle *)objects[i])->n.z;

			minx = MIN(minx, _objects[i].p0.x);
			miny = MIN(miny, _objects[i].p0.y);
			minz = MIN(minz, _objects[i].p0.z);
			maxx = MAX(maxx, _objects[i].p0.x);
			maxy = MAX(maxy, _objects[i].p0.y);
			maxz = MAX(maxz, _objects[i].p0.z);
			minx = MIN(minx, _objects[i].p1.x);
			miny = MIN(miny, _objects[i].p1.y);
			minz = MIN(minz, _objects[i].p1.z);
			maxx = MAX(maxx, _objects[i].p1.x);
			maxy = MAX(maxy, _objects[i].p1.y);
			maxz = MAX(maxz, _objects[i].p1.z);
			minx = MIN(minx, _objects[i].p2.x);
			miny = MIN(miny, _objects[i].p2.y);
			minz = MIN(minz, _objects[i].p2.z);
			maxx = MAX(maxx, _objects[i].p2.x);
			maxy = MAX(maxy, _objects[i].p2.y);
			maxz = MAX(maxz, _objects[i].p2.z);
		}
	}
	
	std::cout << minx << " " << miny << " " << minz << " " << maxx << " " << maxy << " " << maxz << std::endl;

	focal_distance = sqrt(pow((minx + maxx) / 2.0, 2.0) + pow((miny + maxy) / 2.0, 2.0) + pow((minz + maxz) / 2.0, 2.0));
	focal_distance = 575.246;
	std::cout << "focal distance " << focal_distance << std::endl;

	// path tracer initialization
	initializePathTracer(b, current_buffer_host, current_buffer_device, current_doubles_device, _objects, current_objects_device, objects.size(), rand_device, MAX_BOUNCES, WIDTH, HEIGHT, minx, miny, minz, maxx, maxy, maxz);	
}

void enableVerticalSync(bool enableVerticalSync) {
	// WGL_EXT_swap_control.
	typedef BOOL(WINAPI * PFNWGLSWAPINTERVALEXTPROC)(GLint);

	static PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT =
		reinterpret_cast<PFNWGLSWAPINTERVALEXTPROC>(
			wglGetProcAddress("wglSwapIntervalEXT"));

	if (wglSwapIntervalEXT) {
		wglSwapIntervalEXT(enableVerticalSync ? 1 : 0);
		g_enableVerticalSync = enableVerticalSync;
	}
}

void generateCUDAImage() {
	cudaArray *texture_ptr;
	cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0);
	cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0);	
	cudaMemcpyToArray(texture_ptr, 0, 0, fb, fb_size, cudaMemcpyDeviceToDevice);
	cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0);
}