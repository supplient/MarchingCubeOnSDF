#include <cuda_runtime.h>
#include <assert.h>
#include <iostream>

#include "scan.h"
#include "marching_cube.h"

namespace cui {
	void TestInplaceExclusiveScan() {
		// Input data & Allocate host memory
		constexpr int N = 50;
		int h_data[N] = {
			0,1,0,0,1,0,1,0,0,1,
			0,1,0,0,1,0,1,0,0,1,
			0,1,0,0,1,0,1,0,0,1,
			0,1,0,0,1,0,1,0,0,1,
			0,1,0,0,1,0,1,0,0,1,
		};
		int h_std_res[N];
		{
			h_std_res[0] = 0;
			for (int i = 1; i < N; i++)
				h_std_res[i] = h_std_res[i - 1] + h_data[i - 1];
		}

		// Allocate device memory
		int* d_data;
		cudaMalloc(&d_data, N * sizeof(int));
		int* d_res;
		cudaMalloc(&d_res, N * sizeof(int));

		// Send input to GPU
		cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

		// Call the function under test
		InplaceExclusiveScan(d_data, N);

		// Fetch back the result
		cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

		// Assert result
		for (int i = 0; i < N; i++)
			assert(h_data[i] == h_std_res[i]);

		// Free device memory
		cudaFree(d_data);
		cudaFree(d_res);

		// Free host memory
	}

	void TestInplaceInclusiveScan() {
		// Input data & Allocate host memory
		constexpr int N = 50;
		int h_data[N] = {
			0,1,0,0,1,0,1,0,0,1,
			0,1,0,0,1,0,1,0,0,1,
			0,1,0,0,1,0,1,0,0,1,
			0,1,0,0,1,0,1,0,0,1,
			0,1,0,0,1,0,1,0,0,1,
		};
		int h_std_res[N];
		{
			h_std_res[0] = h_data[0];
			for (int i = 1; i < N; i++)
				h_std_res[i] = h_std_res[i - 1] + h_data[i];
		}

		// Allocate device memory
		int* d_data;
		cudaMalloc(&d_data, N * sizeof(int));
		int* d_res;
		cudaMalloc(&d_res, N * sizeof(int));

		// Send input to GPU
		cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

		// Call the function under test
		InplaceInclusiveScan(d_data, N);

		// Fetch back the result
		cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

		// Assert result
		for (int i = 0; i < N; i++)
			assert(h_data[i] == h_std_res[i]);

		// Free device memory
		cudaFree(d_data);
		cudaFree(d_res);

		// Free host memory
	}

	void TestMarchingCube() {
		using namespace std;

		constexpr float R = 0.5;
		constexpr int N = 3; // node number per dim
		constexpr int NODE_NUM = N * N * N;
		constexpr int VOXEL_NUM = (N - 1) * (N - 1) * (N - 1);

		constexpr float low_x = -1.0f;
		constexpr float high_x = 1.0f;
		constexpr float low_y = -1.0f;
		constexpr float high_y = 1.0f;
		constexpr float low_z = -1.0f;
		constexpr float high_z = 1.0f;


		float dx = (high_x - low_x) / float(N - 1);
		float dy = (high_y - low_y) / float(N - 1);
		float dz = (high_z - low_z) / float(N - 1);

		auto CalIndex = [N](int xi, int yi, int zi) {
			return zi * N * N + yi * N + xi;
		};

		float h_sdf[N * N * N];
		Vertex h_positions[N * N * N];
		for (int xi = 0; xi < N; xi++)
			for (int yi = 0; yi < N; yi++)
				for (int zi = 0; zi < N; zi++) {
					float x = low_x + dx * xi;
					float y = low_y + dy * yi;
					float z = low_z + dz * zi;

					float dist = sqrt(x * x + y * y + z * z);
					int id = CalIndex(xi, yi, zi);
					h_sdf[id] = dist - R;
					h_positions[id] = { x, y, z };
				}

		float* d_sdf;
		cudaMalloc(&d_sdf, N * N * N * sizeof(float));
		Vertex* d_positions;
		cudaMalloc(&d_positions, N * N * N * sizeof(Vertex));

		cudaMemcpy(d_sdf, h_sdf, N * N * N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_positions, h_positions, N * N * N * sizeof(Vertex), cudaMemcpyHostToDevice);

		auto res = MarchingCube(d_sdf, d_positions, N, N, N);
		Vertex* d_vert_array = get<0>(res);
		int vert_num = get<1>(res);
		int* d_ind_array = get<2>(res);
		int ind_size = get<3>(res);

		Vertex* h_vert_array = new Vertex[vert_num];
		int* h_ind_array = new int[ind_size];
		cudaMemcpy(h_vert_array, d_vert_array, vert_num * sizeof(Vertex), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_ind_array, d_ind_array, ind_size * sizeof(int), cudaMemcpyDeviceToHost);

		cout << "vert_array: " << endl;
		for (int i = 0; i < vert_num; i++) {
			const Vertex& v = h_vert_array[i];
			cout << i << ": (" << v.x << "," << v.y << "," << v.z << ")\n";
		}
		cout << endl;

		cout << "ind_array: " << endl;
		for (int i = 0; i < ind_size; i++) {
			if (i % 3 == 0)
				cout << "(";
			cout << h_ind_array[i];
			if (i % 3 == 2)
				cout << "), ";
			else
				cout << ",";

			if (i % 15 == 14)
				cout << endl;
		}
		cout << endl;

		cudaFree(d_vert_array);
		cudaFree(d_ind_array);
		cudaFree(d_sdf);
		cudaFree(d_positions);
		delete[] h_vert_array;
		delete[] h_ind_array;
	}
}
