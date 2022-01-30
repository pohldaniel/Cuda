#define _USE_MATH_DEFINES
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <vector>
#include <iostream>
#include <math.h>

#include "device_launch_parameters.h"

#include "pathtracer.h"

__device__ __vector::__vector() : x(0.0), y(0.0), z(0.0) { }
__device__ __vector::__vector(double x, double y, double z) : x(x), y(y), z(z) { }

__device__ double __vector::dot(__vector w) { return this->x*w.x + this->y*w.y + this->z*w.z; }
__device__ double __vector::operator*(__vector w) { return this->x*w.x + this->y*w.y + this->z*w.z; }
__device__ __vector __vector::cross(__vector w) { return __vector(this->y * w.z - this->z * w.y, this->z * w.x - this->x * w.z, this->x * w.y - this->y * w.x); }

__device__ __vector __vector::add(__vector w) { return __vector(this->x + w.x, this->y + w.y, this->z + w.z); }
__device__ __vector __vector::operator+(__vector w) { return __vector(this->x + w.x, this->y + w.y, this->z + w.z); }
__device__ __vector __vector::sub(__vector w) { return __vector(this->x - w.x, this->y - w.y, this->z - w.z); }
__device__ __vector __vector::operator-(__vector w) { return __vector(this->x - w.x, this->y - w.y, this->z - w.z); }
__device__ __vector __vector::mult(double s) { return __vector(s * this->x, s * this->y, s * this->z); }
__device__ __vector __vector::operator*(double s) { return __vector(s * this->x, s * this->y, s * this->z); }

__device__ __vector __vector::h(__vector w) { return __vector(this->x * w.x, this->y * w.y, this->z * w.z); }
__device__ double __vector::length() { return sqrt(this->x*this->x + this->y*this->y + this->z*this->z); }
__device__ __vector __vector::unit() {
	double mag = sqrt(this->x*this->x + this->y*this->y + this->z*this->z);
	return __vector(this->x / mag, this->y / mag, this->z / mag);
}

__device__ bool rayIntersects_device(_bounding_box& b, unsigned short index, __ray r) {
	// contains objects
	if (b.n_objects_device[index] < 1) return false;

	// containment of ray origin
	if (r.origin.x >= b.minx_device[index] && r.origin.x <= b.maxx_device[index] &&
		r.origin.y >= b.miny_device[index] && r.origin.y <= b.maxy_device[index] &&
		r.origin.z >= b.minz_device[index] && r.origin.z <= b.maxz_device[index])
		return true;

	// intsection tests
	if (r.origin.x < b.minx_device[index] && r.direction.x > 0) { // check left face intersection
		double t = (-b.minx_device[index] + r.origin.x) / -r.direction.x;
		//double x = r.origin.x + t * r.direction.x;
		double y = r.origin.y + t * r.direction.y;
		double z = r.origin.z + t * r.direction.z;
		if (y >= b.miny_device[index] && y <= b.maxy_device[index] && z >= b.minz_device[index] && z <= b.maxz_device[index]) return true;
	}
	if (r.origin.x > b.maxx_device[index] && r.direction.x < 0) { // check right face intersection
		double t = (b.maxx_device[index] - r.origin.x) / r.direction.x;
		//double x = r.origin.x + t * r.direction.x;
		double y = r.origin.y + t * r.direction.y;
		double z = r.origin.z + t * r.direction.z;
		if (y >= b.miny_device[index] && y <= b.maxy_device[index] && z >= b.minz_device[index] && z <= b.maxz_device[index]) return true;
	}
	if (r.origin.y < b.miny_device[index] && r.direction.y > 0) { // check bottom face intersection
		double t = (-b.miny_device[index] + r.origin.y) / -r.direction.y;
		double x = r.origin.x + t * r.direction.x;
		//double y = r.origin.y + t * r.direction.y;
		double z = r.origin.z + t * r.direction.z;
		if (x >= b.minx_device[index] && x <= b.maxx_device[index] && z >= b.minz_device[index] && z <= b.maxz_device[index]) return true;
	}
	if (r.origin.y > b.maxy_device[index] && r.direction.y < 0) { // check top face intersection
		double t = (b.maxy_device[index] - r.origin.y) / r.direction.y;
		double x = r.origin.x + t * r.direction.x;
		//double y = r.origin.y + t * r.direction.y;
		double z = r.origin.z + t * r.direction.z;
		if (x >= b.minx_device[index] && x <= b.maxx_device[index] && z >= b.minz_device[index] && z <= b.maxz_device[index]) return true;
	}
	if (r.origin.z < b.minz_device[index] && r.direction.z > 0) { // check rear face intersection
		double t = (-b.minz_device[index] + r.origin.z) / -r.direction.z;
		double x = r.origin.x + t * r.direction.x;
		double y = r.origin.y + t * r.direction.y;
		//double z = r.origin.z + t * r.direction.z;
		if (x >= b.minx_device[index] && x <= b.maxx_device[index] && y >= b.miny_device[index] && y <= b.maxy_device[index]) return true;
	}
	if (r.origin.z > b.maxz_device[index] && r.direction.z < 0) { // check front face intersection
		double t = (b.maxz_device[index] - r.origin.z) / r.direction.z;
		double x = r.origin.x + t * r.direction.x;
		double y = r.origin.y + t * r.direction.y;
		//double z = r.origin.z + t * r.direction.z;
		if (x >= b.minx_device[index] && x <= b.maxx_device[index] && y >= b.miny_device[index] && y <= b.maxy_device[index]) return true;
	}

	// no intersection
	return false;
}

__device__ short intersects_device(_bounding_box& b, int i, __ray r, short hit_list[]) {
	int index = 0,
		count = 1,
		indices[30000];
	indices[index] = 0;

	short hit_count = 0;
	bool found = false;

	while (index < count && index < 30000) {
		i = indices[index++];
		if (rayIntersects_device(b, i, r)) {
			if (b.depth_device[i] == b.max_depth) {
				for (int j = 0; j < b.n_objects_device[i]; j++) {

					short hit = b.objects_device[b.leaf_id_device[i] * 10000 + j];

					found = false;
					for (int l = 0; l < hit_count; l++) {
						if (hit_list[l] == hit) {
							found = true;
							break;
						}
					}

					if (!found) hit_list[hit_count++] = hit;
				}
			}
			else {
				indices[count++] = b.child0_device[i];
				indices[count++] = b.child1_device[i];
			}
		}
	};

	return hit_count;
}

unsigned short build_tree(_bounding_box& b, __object* _objects, int object_count, short parent, int depth, double minx, double miny, double minz, double maxx, double maxy, double maxz) {
	static unsigned short index = 0, leaf_index = 0;
	unsigned short i = index;
	index++;

	double _width = maxx - minx;
	double _height = maxy - miny;
	double _depth = maxz - minz;

	b.depth[i] = depth;
	b.minx[i] = minx; b.miny[i] = miny; b.minz[i] = minz; b.maxx[i] = maxx; b.maxy[i] = maxy; b.maxz[i] = maxz;
	b.child0[i] = b.child1[i] = -1;
	b.id[i] = i;
	b.leaf_id[i] = -1;
	b.parent[i] = parent;
	b.n_objects[i] = 0;

	if (depth == b.max_depth) { // find objects
		int k = leaf_index++;
		b.leaf_id[i] = k;
		int offset = 10000 * k;
		for (int j = 0; j < object_count; j++) {
			if (_objects[j].type == PLANE) {
				// these will be handled separately for now
			}
			else if (_objects[j].type == SPHERE) {
				double x = minx + _width / 2.0;
				double y = miny + _height / 2.0;
				double z = minz + _depth / 2.0;
				double bounding_radius = sqrt(pow(_width / 2.0, 2.0) + pow(_height / 2.0, 2.0) + pow(_depth / 2.0, 2.0));
				double distance = sqrt(pow(_objects[j].center.x - x, 2.0) + pow(_objects[j].center.y - y, 2.0) + pow(_objects[j].center.z - z, 2.0));
				if (_objects[j].radius + bounding_radius >= distance) {
					b.n_objects[i]++;
					b.objects[offset] = j;
					offset++;
				}
			}
			else if (_objects[j].type == TRIANGLE) {
				double sx = (_objects[j].p0.x < _objects[j].p1.x && _objects[j].p0.x < _objects[j].p2.x) ? _objects[j].p0.x : ((_objects[j].p1.x < _objects[j].p2.x) ? _objects[j].p1.x : _objects[j].p2.x);
				double sy = (_objects[j].p0.y < _objects[j].p1.y && _objects[j].p0.y < _objects[j].p2.y) ? _objects[j].p0.y : ((_objects[j].p1.y < _objects[j].p2.y) ? _objects[j].p1.y : _objects[j].p2.y);
				double sz = (_objects[j].p0.z < _objects[j].p1.z && _objects[j].p0.z < _objects[j].p2.z) ? _objects[j].p0.z : ((_objects[j].p1.z < _objects[j].p2.z) ? _objects[j].p1.z : _objects[j].p2.z);
				double bx = (_objects[j].p0.x > _objects[j].p1.x && _objects[j].p0.x > _objects[j].p2.x) ? _objects[j].p0.x : ((_objects[j].p1.x > _objects[j].p2.x) ? _objects[j].p1.x : _objects[j].p2.x);
				double by = (_objects[j].p0.y > _objects[j].p1.y && _objects[j].p0.y > _objects[j].p2.y) ? _objects[j].p0.y : ((_objects[j].p1.y > _objects[j].p2.y) ? _objects[j].p1.y : _objects[j].p2.y);
				double bz = (_objects[j].p0.z > _objects[j].p1.z && _objects[j].p0.z > _objects[j].p2.z) ? _objects[j].p0.z : ((_objects[j].p1.z > _objects[j].p2.z) ? _objects[j].p1.z : _objects[j].p2.z);
				if (sx > maxx) continue;
				if (sy > maxy) continue;
				if (sz > maxz) continue;
				if (bx < minx) continue;
				if (by < miny) continue;
				if (bz < minz) continue;
				b.n_objects[i]++;
				b.objects[offset] = j;
				offset++;
			}
		}
	}
	else { // split along major axis
		unsigned short axis = ((_width >= _height) && (_width >= _depth)) ? 0 : ((_height >= _depth) ? 1 : 2);
		if (axis == 0) {
			b.child0[i] = build_tree(b, _objects, object_count, i, depth + 1, minx, miny, minz, minx + _width / 2.0, maxy, maxz);
			b.child1[i] = build_tree(b, _objects, object_count, i, depth + 1, minx + _width / 2.0, miny, minz, maxx, maxy, maxz);
		}
		else if (axis == 1) {
			b.child0[i] = build_tree(b, _objects, object_count, i, depth + 1, minx, miny, minz, maxx, miny + _height / 2.0, maxz);
			b.child1[i] = build_tree(b, _objects, object_count, i, depth + 1, minx, miny + _height / 2.0, minz, maxx, maxy, maxz);
		}
		else if (axis == 2) {
			b.child0[i] = build_tree(b, _objects, object_count, i, depth + 1, minx, miny, minz, maxx, maxy, minz + _depth / 2.0);
			b.child1[i] = build_tree(b, _objects, object_count, i, depth + 1, minx, miny, minz + _depth / 2.0, maxx, maxy, maxz);
		}
	}
	return i;
}
unsigned short add_back_tree(_bounding_box& b, int i) {

	b.n_objects[i] += (b.child0[i] > -1 ? add_back_tree(b, b.child0[i]) : 0) + (b.child1[i] > -1 ? add_back_tree(b, b.child1[i]) : 0);

	if (b.n_objects[i] < 8 && b.child0[i] > -1 && b.child1[i] > -1) {

		// merge them
		b.depth[i] = b.max_depth;
		b.leaf_id[i] = b.leaf_id[b.child0[i]];

		int n1 = b.n_objects[b.child1[i]];
		int id1 = b.leaf_id[b.child1[i]];
		int n0 = b.n_objects[b.child0[i]];
		int id0 = b.leaf_id[b.child0[i]];
		int n = 0;
		bool found;
		for (int j = 0; j < n1; j++) {
			found = false;
			for (int k = 0; k < n0 + n; k++) if (b.objects[id0 * 10000 + k] == b.objects[id1 * 10000 + j]) found = true;
			if (!found) {
				b.objects[id0 * 10000 + n0 + n] = b.objects[id1 * 10000 + j];
				n++;
			}
		}

		b.n_objects[i] = n0 + n;

		b.child0[i] = b.child1[i] = -1;
	}
	return b.n_objects[i];
}

__device__ __vector sampleRay(_bounding_box b, __ray ray, __object objects[], int object_count, int depth, float* rand_device, int index, int size, int max_bounces) {
	double epsilon = 0.000001;

	__intersection intersect;
	__intersection isect;

	int which = -1;

	int m = max_bounces > 10 ? 10 : max_bounces;
	__vector __a[10], __b[10], sample(0, 0, 0);

	for (int l = 0; l < m; l++) {
		__a[l].x = __a[l].y = __a[l].z = 0;
		__b[l].x = __b[l].y = __b[l].z = 0;
	}

	for (int l = 0; l < m; l++) {

		ray.direction = ray.direction.unit();

		short hit_list[10000];
		short hit_count = intersects_device(b, 0, ray, hit_list);

		intersect.intersects = false;

		for (int j = 0; j < hit_count; j++) {
			int k = hit_list[j];

			isect.intersects = false;

			if (objects[k].type == SPHERE) {

				double a = ray.direction * ray.direction;
				double b = (ray.direction * ray.origin - ray.direction * objects[k].center) * 2.0;
				double c = ray.origin * ray.origin + objects[k].center * objects[k].center - ray.origin * objects[k].center * 2.0 - objects[k].radius * objects[k].radius;
				double det = b * b - 4 * a * c;
				if (det < epsilon) continue;

				double t0 = (-b + sqrt(det)) / (2 * a);
				double t1 = (-b - sqrt(det)) / (2 * a);
				if (t0 < epsilon && t1 < epsilon) continue;

				isect.intersects = true;
				isect.t = t0 < epsilon ? t1 : (t1 < epsilon ? t0 : (t0 < t1 ? t0 : t1));

				isect.ray.origin = ray.origin + ray.direction * isect.t;

				isect.normal = (isect.ray.origin - objects[k].center).unit();

			}
			else if (objects[k].type == TRIANGLE) {

				double den = ray.direction * objects[k].n0;
				if (fabs(den) < epsilon) continue;

				__vector temp = objects[k].p0 - ray.origin;
				double num = temp * objects[k].n0;
				double num_den = num / den;
				if (num_den < epsilon) continue;

				__vector v0 = objects[k].p1 - objects[k].p0;
				__vector v1 = objects[k].p2 - objects[k].p0;
				__vector p = (ray.origin + ray.direction * num_den) - objects[k].p0;

				double pv0 = p*v0;
				double pv1 = p*v1;
				double v0v0 = v0*v0;
				double v0v1 = v0*v1;
				double v1v1 = v1*v1;
				den = v0v0*v1v1 - v0v1*v0v1;
				double s = (pv0*v1v1 - pv1*v0v1) / den;
				double t = (pv1*v0v0 - pv0*v0v1) / den;
				if (s >= 0 && t >= 0 && s + t<1.0) {

					isect.intersects = true;
					isect.t = num_den;

					isect.ray.origin = ray.origin + ray.direction * isect.t;

					isect.normal = objects[k].n0.unit();
				}

			}

			if (isect.intersects) {
				if (!intersect.intersects || isect.t < intersect.t) {
					intersect = isect;
					which = k;
				}
			}
		}
		for (int j = 0; j < object_count; j++) {

			int k = j;

			isect.intersects = false;

			if (objects[k].type == PLANE) {

				double den = ray.direction * objects[k].normal;
				if (fabs(den) < epsilon) continue;

				__vector temp = objects[k].point - ray.origin;
				double num = temp * objects[k].normal;
				double num_den = num / den;
				if (num_den < epsilon) continue;

				isect.intersects = true;
				isect.t = num_den;

				isect.ray.origin = ray.origin + ray.direction * isect.t;

				isect.normal = objects[k].normal.unit();

			}
			else break;

			if (isect.intersects) {
				if (!intersect.intersects || isect.t < intersect.t) {
					intersect = isect;
					which = k;
				}
			}
		}

		if (intersect.intersects) {

			__a[l] = objects[which].emission;
			__b[l] = objects[which].color;

			// reflection for new ray
			intersect.ray.direction = (ray.direction - intersect.normal * (ray.direction * intersect.normal * 2.0)).unit();

			if (objects[which].material == DIFFUSE) { //DIFFUSE

				__vector w = intersect.normal;

				// cosine weighted sampling
				double u1 = rand_device[size*(l + 1) + index + 0];
				double u2 = rand_device[size*(l + 1) + index + 1];
				double r1 = 2 * M_PI * u1;
				double r2 = sqrt(1 - u2);
				double r3 = sqrt(u2);

				__vector u(0, 0, 0);
				if (fabs(w.x) < fabs(w.y) && fabs(w.x) < fabs(w.z)) u.x = 1;
				else if (fabs(w.y) < fabs(w.x) && fabs(w.y) < fabs(w.z)) u.y = 1;
				else u.z = 1;

				u = u.cross(w).unit();
				__vector v = w.cross(u).unit();
				u = v.cross(w).unit();
				__vector d = (u * (cos(r1) * r2) + v * (sin(r1) * r2) + w * r3).unit();

				intersect.ray.direction = d;

				ray = intersect.ray;

			}
			else if (objects[which].material == SPECULAR) { //SPECULAR

				ray = intersect.ray;

			}
			else if (objects[which].material == REFRACTIVE) { //REFRACTIVE

				bool into = ray.direction * intersect.normal < 0; // entering the medium

				double n1n2 = into ? (1.0 / 1.5) : (1.5 / 1.0);
				__vector n = into ? intersect.normal : (intersect.normal * -1);
				__vector r = ray.direction;

				double n1n22 = n1n2 * n1n2;
				double rn = r * n;
				double rn2 = rn * rn;

				double a = 1 - n1n22 * (1 - rn2);
				if (a >= 0) {
					ray.origin = intersect.ray.origin;
					ray.direction = r * n1n2 - n * (n1n2 * rn + sqrt(a));
				}
				else ray = intersect.ray; // total internal reflection

			}

		}
		else break;
	}

	sample = __a[m - 1];
	for (int l = m - 2; l >= 0; l--) sample = __a[l] + __b[l].h(sample);

	return sample;
}

__global__ void kernel(_bounding_box b, float *fb, unsigned char* current_buffer_device, double* current_doubles_device, float* rand_device, __object objects[], int samples, int object_count, int max_bounces, int width, int height, int offsetx, int offsety, double focal_distance, double blur_radius) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	//	int index = y * (width * 3) + x * 3;
	int index = (y + offsety) * (width) + (x + offsetx);

	// cosine weighted sampling
	double u1 = rand_device[index * 3 + 0];
	double u2 = rand_device[index * 3 + 1];
	double r1 = 2 * M_PI * u1;
	double r2 = sqrt(1 - u2);
	double r3 = sqrt(u2);
	__vector offset = __vector(cos(r1)*r2, sin(r1)*r2, r3) * 0.5;

	__vector dir = __vector((x + offsetx) - width / 2, -(y + offsety) + height / 2, 0 + width) + offset;
	__ray ray = { __vector(0, 0, 0), dir.unit() };

	// for depth of field
	u1 = rand_device[index * 3 + 1];
	u2 = rand_device[index * 3 + 2];
	r1 = 2 * M_PI * u1;
	r2 = u2;
	offset = __vector(cos(r1)*r2, sin(r1)*r2, 0.0) * blur_radius;
	__vector p = ray.origin + dir * (focal_distance / width);
	ray.origin = ray.origin + offset;
	ray.direction = (p - ray.origin).unit();

	// sample the ray
	__vector _sample = sampleRay(b, ray, objects, object_count, 0, rand_device, index, width*height * 3, max_bounces);

	// add the sample to the  accumulation
	current_doubles_device[index * 4 + 0] = (current_doubles_device[index * 4 + 0] * (samples - 1.0) + _sample.x) / samples;
	current_doubles_device[index * 4 + 1] = (current_doubles_device[index * 4 + 1] * (samples - 1.0) + _sample.y) / samples;
	current_doubles_device[index * 4 + 2] = (current_doubles_device[index * 4 + 2] * (samples - 1.0) + _sample.z) / samples;

	// save the current frame
	current_buffer_device[index * 4 + 0] = (current_doubles_device[index * 4 + 0] * 255) > 255 ? 255 : (int)(current_doubles_device[index * 4 + 0] * 255);
	current_buffer_device[index * 4 + 1] = (current_doubles_device[index * 4 + 1] * 255) > 255 ? 255 : (int)(current_doubles_device[index * 4 + 1] * 255);
	current_buffer_device[index * 4 + 2] = (current_doubles_device[index * 4 + 2] * 255) > 255 ? 255 : (int)(current_doubles_device[index * 4 + 2] * 255);

	fb[index * 4 + 0] = current_doubles_device[index * 4 + 0];
	fb[index * 4 + 1] = current_doubles_device[index * 4 + 1];
	fb[index * 4 + 2] = current_doubles_device[index * 4 + 2];
	fb[index * 4 + 3] = 1.0;
	//fb[index + 3] = 1.0;
}

extern "C" bool initializePathTracer(_bounding_box& b, unsigned char*& current_buffer_host, unsigned char *& current_buffer_device, double*& current_doubles_device, __object* _objects, __object*& current_objects_device, int object_count, float*& rand_device, int max_bounces, int width, int height, double minx, double miny, double minz, double maxx, double maxy, double maxz) {

	int num_bytes;

	// set up tree -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	b.max_depth = 14;
	b.size = pow(2, b.max_depth + 1) - 1;
	b.leaf_size = pow(2, b.max_depth);

	b.depth = (unsigned short *)malloc(sizeof(unsigned short) * b.size);                 printf("%s\n", cudaGetErrorString(cudaMalloc((void**)&b.depth_device, sizeof(unsigned short) * b.size)));
	b.minx = (double *)malloc(sizeof(double) * b.size);                                  printf("%s\n", cudaGetErrorString(cudaMalloc((void**)&b.minx_device, sizeof(double) * b.size)));
	b.miny = (double *)malloc(sizeof(double) * b.size);                                  printf("%s\n", cudaGetErrorString(cudaMalloc((void**)&b.miny_device, sizeof(double) * b.size)));
	b.minz = (double *)malloc(sizeof(double) * b.size);                                  printf("%s\n", cudaGetErrorString(cudaMalloc((void**)&b.minz_device, sizeof(double) * b.size)));
	b.maxx = (double *)malloc(sizeof(double) * b.size);                                  printf("%s\n", cudaGetErrorString(cudaMalloc((void**)&b.maxx_device, sizeof(double) * b.size)));
	b.maxy = (double *)malloc(sizeof(double) * b.size);                                  printf("%s\n", cudaGetErrorString(cudaMalloc((void**)&b.maxy_device, sizeof(double) * b.size)));
	b.maxz = (double *)malloc(sizeof(double) * b.size);                                  printf("%s\n", cudaGetErrorString(cudaMalloc((void**)&b.maxz_device, sizeof(double) * b.size)));
	b.id = (short *)malloc(sizeof(short) * b.size);                                      printf("%s\n", cudaGetErrorString(cudaMalloc((void**)&b.id_device, sizeof(short) * b.size)));
	b.leaf_id = (short *)malloc(sizeof(short) * b.size);                                 printf("%s\n", cudaGetErrorString(cudaMalloc((void**)&b.leaf_id_device, sizeof(short) * b.size)));
	b.child0 = (short *)malloc(sizeof(short) * b.size);                                  printf("%s\n", cudaGetErrorString(cudaMalloc((void**)&b.child0_device, sizeof(short) * b.size)));
	b.child1 = (short *)malloc(sizeof(short) * b.size);                                  printf("%s\n", cudaGetErrorString(cudaMalloc((void**)&b.child1_device, sizeof(short) * b.size)));
	b.parent = (short *)malloc(sizeof(short) * b.size);                                  printf("%s\n", cudaGetErrorString(cudaMalloc((void**)&b.parent_device, sizeof(short) * b.size)));
	b.n_objects = (unsigned short *)malloc(sizeof(unsigned short) * b.size);             printf("%s\n", cudaGetErrorString(cudaMalloc((void**)&b.n_objects_device, sizeof(unsigned short) * b.size)));
	b.objects = (unsigned short *)malloc(sizeof(unsigned short) * b.leaf_size * 10000);  printf("%s\n", cudaGetErrorString(cudaMalloc((void**)&b.objects_device, sizeof(unsigned short) * b.leaf_size * 10000)));

	build_tree(b, _objects, object_count, -1, 0, minx, miny, minz, maxx, maxy, maxz);
	add_back_tree(b, 0);

	printf("%s\n", cudaGetErrorString(cudaMemcpy(b.depth_device, b.depth, sizeof(unsigned short) * b.size, cudaMemcpyHostToDevice)));
	printf("%s\n", cudaGetErrorString(cudaMemcpy(b.minx_device, b.minx, sizeof(double) * b.size, cudaMemcpyHostToDevice)));
	printf("%s\n", cudaGetErrorString(cudaMemcpy(b.miny_device, b.miny, sizeof(double) * b.size, cudaMemcpyHostToDevice)));
	printf("%s\n", cudaGetErrorString(cudaMemcpy(b.minz_device, b.minz, sizeof(double) * b.size, cudaMemcpyHostToDevice)));
	printf("%s\n", cudaGetErrorString(cudaMemcpy(b.maxx_device, b.maxx, sizeof(double) * b.size, cudaMemcpyHostToDevice)));
	printf("%s\n", cudaGetErrorString(cudaMemcpy(b.maxy_device, b.maxy, sizeof(double) * b.size, cudaMemcpyHostToDevice)));
	printf("%s\n", cudaGetErrorString(cudaMemcpy(b.maxz_device, b.maxz, sizeof(double) * b.size, cudaMemcpyHostToDevice)));
	printf("%s\n", cudaGetErrorString(cudaMemcpy(b.id_device, b.id, sizeof(short) * b.size, cudaMemcpyHostToDevice)));
	printf("%s\n", cudaGetErrorString(cudaMemcpy(b.leaf_id_device, b.leaf_id, sizeof(short) * b.size, cudaMemcpyHostToDevice)));
	printf("%s\n", cudaGetErrorString(cudaMemcpy(b.child0_device, b.child0, sizeof(short) * b.size, cudaMemcpyHostToDevice)));
	printf("%s\n", cudaGetErrorString(cudaMemcpy(b.child1_device, b.child1, sizeof(short) * b.size, cudaMemcpyHostToDevice)));
	printf("%s\n", cudaGetErrorString(cudaMemcpy(b.parent_device, b.parent, sizeof(short) * b.size, cudaMemcpyHostToDevice)));
	printf("%s\n", cudaGetErrorString(cudaMemcpy(b.n_objects_device, b.n_objects, sizeof(unsigned short) * b.size, cudaMemcpyHostToDevice)));
	printf("%s\n", cudaGetErrorString(cudaMemcpy(b.objects_device, b.objects, sizeof(unsigned short) * b.leaf_size * 10000, cudaMemcpyHostToDevice)));
	// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	// for storing the image result
	num_bytes = sizeof(unsigned char) * width * height * 4;
	current_buffer_host = (unsigned char*)malloc(num_bytes);
	cudaMalloc((void**)&(current_buffer_device), num_bytes);
	cudaMemset(current_buffer_device, 0, num_bytes);

	// for accumulating samples
	num_bytes = sizeof(double) * width * height * 4;
	cudaMalloc((void**)&(current_doubles_device), num_bytes);
	cudaMemset(current_doubles_device, 0, num_bytes);

	// for generating random uniforms
	int m = 1 + max_bounces;//4 + max_bounces * 4;
	num_bytes = sizeof(float) * width * height * 3 * m;
	cudaMalloc((void**)&rand_device, num_bytes);

	// for storing the objects on the device
	num_bytes = sizeof(__object) * object_count;
	cudaMalloc((void**)&current_objects_device, num_bytes);
	cudaMemcpy(current_objects_device, _objects, num_bytes, cudaMemcpyHostToDevice);

	return true;
}

extern "C" bool setupPathTracer(double*& current_doubles_device, __object*& current_objects_device, __object* _objects, int object_count, int width, int height) {
	int num_bytes = sizeof(double) * width * height * 4;

	// zero out the sample accumulation
	cudaMemset(current_doubles_device, 0, num_bytes);

	// store the objects on the device
	num_bytes = sizeof(__object) * object_count;
	cudaMemcpy(current_objects_device, _objects, num_bytes, cudaMemcpyHostToDevice);

	return true;
}

extern "C" bool releasePathTracer(_bounding_box& b, unsigned char*& current_buffer_host, unsigned char *& current_buffer_device, double*& current_doubles_device, __object*& current_objects_device, float*& rand_device, int width, int height) {

	free(b.depth);     cudaFree(b.depth_device);
	free(b.minx);      cudaFree(b.minx_device);
	free(b.miny);      cudaFree(b.miny_device);
	free(b.minz);      cudaFree(b.minz_device);
	free(b.maxx);      cudaFree(b.maxx_device);
	free(b.maxy);      cudaFree(b.maxy_device);
	free(b.maxz);      cudaFree(b.maxz_device);
	free(b.id);        cudaFree(b.id_device);
	free(b.leaf_id);   cudaFree(b.leaf_id_device);
	free(b.child0);    cudaFree(b.child0_device);
	free(b.child1);    cudaFree(b.child1_device);
	free(b.parent);    cudaFree(b.parent_device);
	free(b.n_objects); cudaFree(b.n_objects_device);
	free(b.objects);   cudaFree(b.objects_device);

	free(current_buffer_host);
	cudaFree(current_buffer_device);

	cudaFree(current_doubles_device);

	cudaFree(rand_device);

	cudaFree(current_objects_device);

	return true;
}

extern "C" bool runPathTracer(_bounding_box& b, float *fb, unsigned char*& current_buffer_host, unsigned char *& current_buffer_device, double*& current_doubles_device, float*& rand_device, unsigned long long samples, __object objects[], __object*& current_objects_device, int object_count, int max_bounces, int width, int height, int offsetx, int offsety, double focal_distance, double blur_radius) {

	int dimx = 16;
	int dimy = 16;
	dim3 dimGrid(4, 4);
	//	dim3 dimGrid(width/dimx, height/dimy);
	dim3 dimBlock(dimx, dimy);

	// generate random uniforms
	if (offsetx == 0 && offsety == 0) {
		int m = 1 + max_bounces;//4 + max_bounces * 4; // 1 plus 10 depth
		curandGenerator_t gen;
		curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
		curandSetPseudoRandomGeneratorSeed(gen, samples); //samples=seed
		curandGenerateUniform(gen, rand_device, width * height * 3 * m);
		curandDestroyGenerator(gen);
	}

	//cudaDeviceSetLimit (cudaLimitStackSize, );
	//cudaThreadSetLimit (cudaLimitStackSize, 8192*16)

	kernel << <dimGrid, dimBlock >> >(b, fb, current_buffer_device, current_doubles_device, rand_device, current_objects_device, samples, object_count, max_bounces, width, height, offsetx, offsety, focal_distance, blur_radius);

	int num_bytes = sizeof(unsigned char) * width * height * 4;
	printf("%s\n", cudaGetErrorString(cudaMemcpy(current_buffer_host, current_buffer_device, num_bytes, cudaMemcpyDeviceToHost)));

	return true;
}