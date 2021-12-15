#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ray.h"
#include "vec3.h"
#include "hitable.h"
#include "hitable_list.h"
#include "sphere.h"

__device__ vec3 color(const ray& r, hitable **world) {
	hit_record rec;
	if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
		return 0.5f*vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
	}
	else {
		vec3 unit_direction = unit_vector(r.direction());
		float t = 0.5f*(unit_direction.y() + 1.0f);
		return (1.0f - t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
	}
}

__global__ void _create_world(hitable **d_list, hitable **d_world) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*(d_list) = new sphere(vec3(0, 0, -1), 0.5);
		*(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100);
		*d_world = new hitable_list(d_list, 2);
	}
}

__global__ void free_world(hitable **d_list, hitable **d_world) {
	delete *(d_list);
	delete *(d_list + 1);
	delete *d_world;
}

__global__ void render(float *fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin, hitable **world) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x * 4 + i * 4;
	float u = float(i) / float(max_x);
	float v = float(j) / float(max_y);
	ray r(origin, lower_left_corner + u*horizontal + v*vertical);
	vec3 c = color(r, world);

	fb[pixel_index + 0] = c.r();
	fb[pixel_index + 1] = c.g();
	fb[pixel_index + 2] = c.b();
	fb[pixel_index + 3] = 1.0;
}

extern "C" void create_image(dim3 blocks, dim3 threads, float *fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin, hitable **world) {
	render << <blocks, threads >> >(fb, max_x, max_y, lower_left_corner, horizontal, vertical, origin, world);
}


extern "C" void create_world(hitable **list, hitable **world) {
	_create_world << <1, 1 >> >(list, world);
}