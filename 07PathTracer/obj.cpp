#include "obj.h"

cObj::cObj(std::string filename) {
	std::ifstream ifs(filename.c_str(), std::ifstream::in);
	std::string line, key;
	while (ifs.good() && !ifs.eof() && std::getline(ifs, line)) {
		key = "";
		std::stringstream stringstream(line);
		stringstream >> key >> std::ws;

		if (key == "v") { // vertex
			vertex v; float x, y, z;
			while (!stringstream.eof()) {
				stringstream >> x >> std::ws >> y >> std::ws >> z >> std::ws;
				minx = minx < x ? minx : x;
				miny = miny < y ? miny : y;
				minz = minz < z ? minz : z;
				maxx = maxx > x ? maxx : x;
				maxy = maxy > y ? maxy : y;
				maxz = maxz > z ? maxz : z;
				v.v.push_back(x);
				v.v.push_back(y);
				v.v.push_back(z);
			}
			/*while (!stringstream.eof()) {
			stringstream >> x >> std::ws;
			v.v.push_back(x);
			}*/
			vertices.push_back(v);
		}
		else if (key == "vp") { // parameter
			vertex v; float x;
			while (!stringstream.eof()) {
				stringstream >> x >> std::ws;
				v.v.push_back(x);
			}
			parameters.push_back(v);
		}
		else if (key == "vt") { // texture coordinate
			vertex v; float x;
			while (!stringstream.eof()) {
				stringstream >> x >> std::ws;
				v.v.push_back(x);
			}
			texcoords.push_back(v);
		}
		else if (key == "vn") { // normal
			vertex v; float x;
			while (!stringstream.eof()) {
				stringstream >> x >> std::ws;
				v.v.push_back(x);
			}
			v.normalize();
			normals.push_back(v);
		}
		else if (key == "f") { // face
			face f; int v, t, n;
			while (!stringstream.eof()) {
				stringstream >> v >> std::ws;
				f.vertex.push_back(v - 1);
				if (stringstream.peek() == '/') {
					stringstream.get();
					if (stringstream.peek() == '/') {
						stringstream.get();
						stringstream >> n >> std::ws;
						f.normal.push_back(n - 1);
					}
					else {
						stringstream >> t >> std::ws;
						f.texture.push_back(t - 1);
						if (stringstream.peek() == '/') {
							stringstream.get();
							stringstream >> n >> std::ws;
							f.normal.push_back(n - 1);
						}
					}
				}
			}
			faces.push_back(f);
		}
		else {

		}
	}
	ifs.close();
	std::cout << "               Name: " << filename << std::endl;
	std::cout << "           Vertices: " << number_format(vertices.size()) << std::endl;
	std::cout << "         Parameters: " << number_format(parameters.size()) << std::endl;
	std::cout << "Texture Coordinates: " << number_format(texcoords.size()) << std::endl;
	std::cout << "            Normals: " << number_format(normals.size()) << std::endl;
	std::cout << "              Faces: " << number_format(faces.size()) << std::endl << std::endl;
	//	list = glGenLists(1);
	//	compileList();
	//	vertices.clear();
	//	texcoords.clear();
	//	normals.clear();
	//	faces.clear();
}

cObj::~cObj() {
	//	glDeleteLists(list, 1);
	vertices.clear();
	texcoords.clear();
	normals.clear();
	faces.clear();
	parameters.clear();
}

void cObj::compileList() {
	glNewList(list, GL_COMPILE);
	for (int i = 0; i < faces.size(); i++) {
		if (faces[i].vertex.size() == 3) { // triangle
			if (faces[i].normal.size() == 3) { // with normals
				glBegin(GL_TRIANGLES);
				glNormal3f(normals[faces[i].normal[0]].v[0], normals[faces[i].normal[0]].v[1], normals[faces[i].normal[0]].v[2]);
				glVertex3f(vertices[faces[i].vertex[0]].v[0], vertices[faces[i].vertex[0]].v[1], vertices[faces[i].vertex[0]].v[2]);
				glNormal3f(normals[faces[i].normal[1]].v[0], normals[faces[i].normal[1]].v[1], normals[faces[i].normal[1]].v[2]);
				glVertex3f(vertices[faces[i].vertex[1]].v[0], vertices[faces[i].vertex[1]].v[1], vertices[faces[i].vertex[1]].v[2]);
				glNormal3f(normals[faces[i].normal[2]].v[0], normals[faces[i].normal[2]].v[1], normals[faces[i].normal[2]].v[2]);
				glVertex3f(vertices[faces[i].vertex[2]].v[0], vertices[faces[i].vertex[2]].v[1], vertices[faces[i].vertex[2]].v[2]);
				glEnd();
			}
			else { // without normals -- evaluate normal on quad
				vertex v = (vertices[faces[i].vertex[1]] - vertices[faces[i].vertex[0]]).cross(vertices[faces[i].vertex[2]] - vertices[faces[i].vertex[0]]);
				v.normalize();
				glBegin(GL_TRIANGLES);
				glNormal3f(v.v[0], v.v[1], v.v[2]);
				glVertex3f(vertices[faces[i].vertex[0]].v[0], vertices[faces[i].vertex[0]].v[1], vertices[faces[i].vertex[0]].v[2]);
				glVertex3f(vertices[faces[i].vertex[1]].v[0], vertices[faces[i].vertex[1]].v[1], vertices[faces[i].vertex[1]].v[2]);
				glVertex3f(vertices[faces[i].vertex[2]].v[0], vertices[faces[i].vertex[2]].v[1], vertices[faces[i].vertex[2]].v[2]);
				glEnd();
			}
		}
		else if (faces[i].vertex.size() == 4) { // quad
			if (faces[i].normal.size() == 4) { // with normals
				glBegin(GL_QUADS);
				glNormal3f(normals[faces[i].normal[0]].v[0], normals[faces[i].normal[0]].v[1], normals[faces[i].normal[0]].v[2]);
				glVertex3f(vertices[faces[i].vertex[0]].v[0], vertices[faces[i].vertex[0]].v[1], vertices[faces[i].vertex[0]].v[2]);
				glNormal3f(normals[faces[i].normal[1]].v[0], normals[faces[i].normal[1]].v[1], normals[faces[i].normal[1]].v[2]);
				glVertex3f(vertices[faces[i].vertex[1]].v[0], vertices[faces[i].vertex[1]].v[1], vertices[faces[i].vertex[1]].v[2]);
				glNormal3f(normals[faces[i].normal[2]].v[0], normals[faces[i].normal[2]].v[1], normals[faces[i].normal[2]].v[2]);
				glVertex3f(vertices[faces[i].vertex[2]].v[0], vertices[faces[i].vertex[2]].v[1], vertices[faces[i].vertex[2]].v[2]);
				glNormal3f(normals[faces[i].normal[3]].v[0], normals[faces[i].normal[3]].v[1], normals[faces[i].normal[3]].v[2]);
				glVertex3f(vertices[faces[i].vertex[3]].v[0], vertices[faces[i].vertex[3]].v[1], vertices[faces[i].vertex[3]].v[2]);
				glEnd();
			}
			else { // without normals -- evaluate normal on quad
				vertex v = (vertices[faces[i].vertex[1]] - vertices[faces[i].vertex[0]]).cross(vertices[faces[i].vertex[2]] - vertices[faces[i].vertex[0]]);
				v.normalize();
				glBegin(GL_QUADS);
				glNormal3f(v.v[0], v.v[1], v.v[2]);
				glVertex3f(vertices[faces[i].vertex[0]].v[0], vertices[faces[i].vertex[0]].v[1], vertices[faces[i].vertex[0]].v[2]);
				glVertex3f(vertices[faces[i].vertex[1]].v[0], vertices[faces[i].vertex[1]].v[1], vertices[faces[i].vertex[1]].v[2]);
				glVertex3f(vertices[faces[i].vertex[2]].v[0], vertices[faces[i].vertex[2]].v[1], vertices[faces[i].vertex[2]].v[2]);
				glVertex3f(vertices[faces[i].vertex[3]].v[0], vertices[faces[i].vertex[3]].v[1], vertices[faces[i].vertex[3]].v[2]);
				glEnd();
			}
		}
	}
	glEndList();
}

void cObj::render() {
	glCallList(list);
}