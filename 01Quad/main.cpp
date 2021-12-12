#define NOMINMAX
#define STB_IMAGE_IMPLEMENTATION
#include <windows.h>
#include <stdio.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#include "stb\stb_image.h"
#include "Extension.h"
#include "Quad.h"

unsigned int window_width = 512;
unsigned int window_height = 512;
unsigned int image_width = 512;
unsigned int image_height = 512;

Quad *quad;
GLuint m_texture;
unsigned int *cuda_dest_resource;
struct cudaGraphicsResource *cuda_tex_result_resource;

POINT g_OldCursorPos;
bool g_enableVerticalSync;

LRESULT CALLBACK winProc(HWND hWnd, UINT message, WPARAM wParma, LPARAM lParam);
void initApp(HWND hWnd);
void enableVerticalSync(bool enableVerticalSync);
void generateCUDAImage();
////////////////////////////////////////////////////////////////////////////////
extern "C" void launch_cudaProcess(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int imgw);

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
		"Simple CUDA",							// app name
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT,			// x,y coordinate
		window_width,
		window_height,							// width, height
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
			generateCUDAImage();

			glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			
			quad->render(m_texture);

			hdc = GetDC(hwnd);
			SwapBuffers(hdc);
			ReleaseDC(hwnd, hdc);
		}
	} // end while
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

	/*stbi_set_flip_vertically_on_load(true);
	int width, height, numCompontents;
	unsigned char* imageData = stbi_load("test.png", &width, &height, &numCompontents, 4);

	// create a texture
	glGenTextures(1, &m_texture);
	glBindTexture(GL_TEXTURE_2D, m_texture);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageData);
	glGenerateMipmap(GL_TEXTURE_2D);
	stbi_image_free(imageData);*/


	// create a texture
	glGenTextures(1, &m_texture);
	glBindTexture(GL_TEXTURE_2D, m_texture);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	// register this texture with CUDA
	cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, m_texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);

	//Init CUDABuffer
	unsigned int num_texels = image_width * image_height;
	unsigned int num_values = num_texels * 4;
	unsigned int size_tex_data = sizeof(GLubyte) * num_values;
	cudaMalloc((void **)&cuda_dest_resource, size_tex_data);
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

	dim3 block(16, 16, 1);
	dim3 grid(image_width / block.x, image_height / block.y, 1);

	launch_cudaProcess(grid, block, 0, cuda_dest_resource, image_width);

	cudaArray *texture_ptr;
	cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0);
	cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0);

	int num_texels = image_width * image_height;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;
	cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource, size_tex_data, cudaMemcpyDeviceToDevice);
	cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0);
}
