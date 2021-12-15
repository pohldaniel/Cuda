#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ray.h"
#include "vec3.h"

__device__ bool hit_sphere(const vec3& center, float radius, const ray& r) {
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = 2.0f * dot(oc, r.direction());
	float c = dot(oc, oc) - radius*radius;
	float discriminant = b*b - 4.0f*a*c;
	return (discriminant > 0.0f);
}

__device__ vec3 color(const ray& r) {
	if (hit_sphere(vec3(0, 0, -1), 0.5, r))
		return vec3(1, 0, 0);
	vec3 unit_direction = unit_vector(r.direction());
	float t = 0.5f*(unit_direction.y() + 1.0f);
	return (1.0f - t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

__global__ void render(float *fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x * 4 + i * 4;
	float u = float(i) / float(max_x);
	float v = float(j) / float(max_y);
	ray r(origin, lower_left_corner + u*horizontal + v*vertical);
	vec3 c = color(r);

	fb[pixel_index + 0] = c.r();
	fb[pixel_index + 1] = c.g();
	fb[pixel_index + 2] = c.b();
	fb[pixel_index + 3] = 1.0;
}

extern "C" void launch_cudaProcess(dim3 blocks, dim3 threads, float *fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {
	render << <blocks, threads >> >(fb, max_x, max_y, lower_left_corner, horizontal, vertical, origin);
}
