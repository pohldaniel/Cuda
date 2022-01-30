#ifndef PATHTRACERH
#define PATHTRACERH

class __vector {
public:
	double x, y, z;

	__device__ __vector();
	__device__ __vector(double x, double y, double z);

	__device__ double dot(__vector w);
	__device__ double operator*(__vector w);
	__device__ __vector cross(__vector w);

	__device__ __vector add(__vector w);
	__device__ __vector operator+(__vector w);
	__device__ __vector sub(__vector w);
	__device__ __vector operator-(__vector w);
	__device__ __vector mult(double s);
	__device__ __vector operator*(double s);

	__device__ __vector h(__vector w);
	__device__ double length();
	__device__ __vector unit();
};


struct __ray {
	__vector origin;
	__vector direction;
};

struct __intersection {
	bool intersects;
	double t;

	__ray ray;
	__vector normal;
};

struct __object {
	int type; // geometry
	int material;
	__vector color;
	__vector emission;

	// for planes
	__vector normal;
	__vector point;

	// for spheres
	__vector center;
	double radius;

	// for triangles
	__vector p0, p1, p2;
	__vector n0, n1, n2;
	__vector n;
};

struct _bounding_box {
	unsigned short *depth,
		*depth_device;
	double *minx, *miny, *minz, *maxx, *maxy, *maxz,
		*minx_device, *miny_device, *minz_device, *maxx_device, *maxy_device, *maxz_device;
	short *child0, *child1,
		*child0_device, *child1_device;
	short *parent,
		*parent_device;

	short *id, *id_device;
	short *leaf_id, *leaf_id_device;

	unsigned short *n_objects,
		*n_objects_device;
	unsigned short *objects,
		*objects_device;
	unsigned short max_depth;
	unsigned short size, leaf_size;
};

enum geometry { NONE, PLANE, SPHERE, TRIANGLE };
enum materials { DIFFUSE, SPECULAR, REFRACTIVE };

#endif