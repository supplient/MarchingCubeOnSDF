#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helper.h"

using namespace std;

namespace cui {
	namespace exclusive_scan {
		__global__ void kernel_InplaceReduce(int* data, int N) {
			int bi = blockIdx.x;
			int ti = threadIdx.x;

			int offset = bi * blockDim.x * 2;
			N -= bi * blockDim.x * 2;
			if (N > blockDim.x * 2)
				N = blockDim.x * 2;

			int di = (2 * ti + 1) - (blockDim.x * 2 - N);
			if (di < 0)
				return;
			for (int stride = 1; stride <= blockDim.x; stride *= 2) {
				__syncthreads();
				if ((ti + 1) % stride != 0)
					break;
				int neighbori = di - stride;
				if (neighbori < 0)
					continue;
				data[offset + di] += data[offset + neighbori];
			}
		}

		__global__ void kernel_InplaceDownSweep(int* data, int N) {
			int bi = blockIdx.x;
			int ti = threadIdx.x;

			int offset = bi * blockDim.x * 2;
			N -= bi * blockDim.x * 2;
			if (N > blockDim.x * 2)
				N = blockDim.x * 2;

			int di = (2 * ti + 1) - (blockDim.x * 2 - N);
			if (di < 0)
				return;

			if (ti == blockDim.x - 1)
				data[offset + di] = 0;

			for (int stride = blockDim.x; stride >= 1; stride /= 2) {
				__syncthreads();
				if ((ti + 1) % stride != 0)
					continue;
				int neighbori = di - stride;
				if (neighbori < 0)
					continue;
				int tmp = data[offset + neighbori];
				data[offset + neighbori] = data[offset + di];
				data[offset + di] += tmp;
			}
		}

		__global__ void kernel_CopyToSums(int* data, int* sums, int N, int per_block) {
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			int data_id = (i + 1) * per_block - 1;
			if (data_id >= N)
				data_id = N - 1;
			sums[i] = data[data_id];
		}

		__global__ void kernel_AddToEachBlock(int* data, int* sums, int N) {
			int bi = blockIdx.x;
			int ti = threadIdx.x;
			int di = (bi * blockDim.x + ti) * 2;

			if (di < N) {
				data[di] += sums[bi];
				data[di + 1] += sums[bi];
			}
		}

	}

	namespace inclusive_scan {
		__global__ void kernel_AddBackup(int* data, int* backup, int N) {
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			if (i < N)
				data[i] += backup[i];
		}
	}



	void InplaceExclusiveScan(int* d_data, int N) {
		//
		// refer to https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
		//
		using namespace exclusive_scan;

		constexpr int BLOCK_SIZE = 32; // Must be pow of 2
		int block_num = CalBlockNum(N, 2 * BLOCK_SIZE);
		if (block_num > 2 * BLOCK_SIZE)
			throw "ExclusiveScan Error: BLOCK_SIZE too small, cannot afford such large array.";

		kernel_InplaceReduce << <block_num, BLOCK_SIZE >> > (d_data, N);

		int* d_sums;
		const int& sum_num = block_num; // just an alias
		cudaMalloc(&d_sums, sum_num * sizeof(int));
		kernel_CopyToSums << <CalBlockNum(sum_num, BLOCK_SIZE), BLOCK_SIZE >> > (
			d_data, d_sums,
			N, 2 * BLOCK_SIZE
			);
		kernel_InplaceReduce << <CalBlockNum(sum_num, 2 * BLOCK_SIZE), BLOCK_SIZE >> > (d_sums, sum_num);
		kernel_InplaceDownSweep << <CalBlockNum(sum_num, 2 * BLOCK_SIZE), BLOCK_SIZE >> > (d_sums, sum_num);

		kernel_InplaceDownSweep << <block_num, BLOCK_SIZE >> > (d_data, N);
		kernel_AddToEachBlock << <CalBlockNum(N, 2 * BLOCK_SIZE), BLOCK_SIZE >> > (d_data, d_sums, N);
		cudaFree(d_sums);
	}

	void InplaceInclusiveScan(int* d_data, int N) {
		using namespace inclusive_scan;

		constexpr int BLOCK_SIZE = 32;
		int* d_data_backup;
		cudaMalloc(&d_data_backup, N * sizeof(int));
		cudaMemcpy(d_data_backup, d_data, N * sizeof(int), cudaMemcpyDeviceToDevice);
		InplaceExclusiveScan(d_data, N);
		kernel_AddBackup << <CalBlockNum(N, BLOCK_SIZE), BLOCK_SIZE >> > (d_data, d_data_backup, N);
		cudaFree(d_data_backup);
	}

};
