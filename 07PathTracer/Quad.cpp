#include "Quad.h"

Quad::Quad(float size, float sizeTex) {
	m_quadShader = new Shader("shader/quad.vs", "shader/quad.fs");
	m_size = size;
	m_sizeTex = sizeTex;
	createBuffer();
}

Quad::~Quad() {

}

void Quad::createBuffer() {

	m_vertex.push_back(1.0 * m_size); m_vertex.push_back(1.0 * m_size); m_vertex.push_back(0.0 * m_size); m_vertex.push_back(0.0 * m_sizeTex); m_vertex.push_back(0.0 * m_sizeTex);
	
	
	m_vertex.push_back(-1.0 * m_size); m_vertex.push_back(1.0 * m_size); m_vertex.push_back(0.0 * m_size); m_vertex.push_back(1.0 * m_sizeTex); m_vertex.push_back(0.0 * m_sizeTex);
	m_vertex.push_back(-1.0 * m_size); m_vertex.push_back(-1.0 * m_size); m_vertex.push_back(0.0 * m_size); m_vertex.push_back(1.0 * m_sizeTex); m_vertex.push_back(1.0 * m_sizeTex);
	m_vertex.push_back(1.0 * m_size); m_vertex.push_back(-1.0 * m_size); m_vertex.push_back(0.0 * m_size); m_vertex.push_back(0.0 * m_sizeTex); m_vertex.push_back(1.0 * m_sizeTex);

	static const GLushort index[] = {
		0, 1, 2,
		0, 2, 3
	};

	unsigned int quadVBO, indexQuad;

	glGenVertexArrays(1, &m_vao);
	glBindVertexArray(m_vao);

	glGenBuffers(1, &quadVBO);
	glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
	glBufferData(GL_ARRAY_BUFFER, m_vertex.size() * sizeof(float), &m_vertex[0], GL_STATIC_DRAW);

	//Position
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);

	//Texture Coordinates
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

	//Indices
	glGenBuffers(1, &indexQuad);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexQuad);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(index), index, GL_STATIC_DRAW);

	glBindVertexArray(0);

}

void Quad::render(unsigned int texture) {

	glUseProgram(m_quadShader->m_program);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glBindVertexArray(m_vao);
	glDrawElements(GL_TRIANGLES, 2 * 3, GL_UNSIGNED_SHORT, 0);
	glBindVertexArray(0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);	
	glUseProgram(0);
}

Shader*  Quad::getShader() const {
	return m_quadShader;
}