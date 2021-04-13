#pragma once
#include <tuple>

namespace cui {
	struct Vertex {
		float x;
		float y;
		float z;
	};

	struct VerticesOnEdge {
		Vertex x; // 0
		Vertex y; // 1
		Vertex z; // 2
	};

	typedef int TrianglesInVoxel[5][3];

	std::tuple<Vertex*, int, int*, int> MarchingCube(float* SDF, Vertex* positions, int xn, int yn, int zn);
}
