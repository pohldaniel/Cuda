#ifndef __quadH__
#define __quadH__

#include <vector>

#include "Shader.h"


class Quad {

public:

	Quad(float size = 1.0f, float sizeTex = 1.0f);
	~Quad();

	void render(unsigned int texture);
	Shader* getShader() const;

private:

	void createBuffer();
	unsigned int m_vao;
	float m_size = 1;
	float m_sizeTex = 1;
	std::vector<float> m_vertex;

	Shader *m_quadShader;
};


#endif // __quadH__