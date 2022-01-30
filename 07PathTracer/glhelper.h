#ifndef GLHELPER_H
#define GLHELPER_H

#include <math.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

#include "Extension.h"
#include "misc.h"

//SDL_Surface* mySDLInit(const int WIDTH, const int HEIGHT, const int BPP, const bool fullscreen);

void setupOrtho(int width, int height);

void setupTexture(GLuint& texture);
//void setupTexture(GLuint& texture, SDL_Surface *s);
void setupTextureFloat(GLuint& texture, int w, int h, const float texturef[]);
void setupTextureFloat32(GLuint& texture, int w, int h, const float texturef[]);
void setupTextureRGBA(GLuint& texture, int w, int h, const unsigned char texture_[]);
void setupTextureRGB(GLuint& texture, int w, int h, const unsigned char texture_[]);
void setupTextureImage(GLuint& texture, int w, int h, int bpp, const unsigned char texture_[]);
void setupTextureTGA(GLuint& texture, const char* tga_file, unsigned char*& buffer, int& width, int& height, int& bpp);
void deleteTexture(GLuint& texture);
void deleteTextureTGA(GLuint& texture, unsigned char*& buffer);

void setupCubeMap(GLuint& texture);
//void setupCubeMap(GLuint& texture, SDL_Surface *xpos, SDL_Surface *xneg, SDL_Surface *ypos, SDL_Surface *yneg, SDL_Surface *zpos, SDL_Surface *zneg);
void deleteCubeMap(GLuint& texture);

void createProgram(GLuint& glProgram, GLuint& glShaderV, GLuint& glShaderF, const char* vertex_shader, const char* fragment_shader);
void releaseProgram(GLuint& glProgram, GLuint glShaderV, GLuint glShaderF);

void saveTGA(unsigned char* buffer, int width, int height, bool video = false);
void saveTGARGBA(unsigned char* buffer, int width, int height, bool video);
void saveTGADouble(double* buffer, int width, int height);

void glerror(const char* prepend);

#endif
