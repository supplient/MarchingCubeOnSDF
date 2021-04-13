#pragma once

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "config.h"
#include "helper.h"
#include "scan.h"
#include "cuda_temp_mem.h"


namespace cui {

	namespace stream_compression {
		template <typename InputType, typename CheckFunc>
		__global__ void kernel_Transform(int* stencil, InputType* A, int N, CheckFunc func) {
			int id = blockDim.x * blockIdx.x + threadIdx.x;
			if (id < N)
				stencil[id] = func(A[id]);
		}

		template<typename InputType>
		__global__ void kernel_Copy(InputType* dst, InputType* src, int N) {
			int id = blockDim.x * blockIdx.x + threadIdx.x;
			if (id < N)
				dst[id] = src[id];
		}

		template<typename InputType>
		__global__ void kernel_MapIfLeftDiff(InputType* dst, InputType* src, int* stencil, int N) {
			int id = blockDim.x * blockIdx.x + threadIdx.x;
			if (id < N) {
				if (id == 0 || stencil[id - 1] != stencil[id]) {
					dst[stencil[id] - 1] = src[id];
				}
			}
		}
	}


	template <typename InputType, typename CheckFunc>
	int InplaceStreamCompression(InputType* d_data, int N, CheckFunc func) {
		using namespace stream_compression;

		int block_size = Config::block_size;

		CudaTempMem<int> d_stencil(N);
		kernel_Transform << <CalBlockNum(N, block_size), block_size >> > (
			d_stencil, d_data, N, func
			);
		InplaceInclusiveScan(d_stencil, N);

		int res_num = 0;
		cudaMemcpy(&res_num, d_stencil.Get() + N - 1, sizeof(int), cudaMemcpyDeviceToHost);

		CudaTempMem<int> d_copy(N);
		kernel_Copy << <CalBlockNum(N, block_size), block_size >> > (
			d_copy.Get(), d_data, N
			);

		kernel_MapIfLeftDiff << <CalBlockNum(N, block_size), block_size >> > (
			d_data, d_copy.Get(), d_stencil.Get(), N
			);

		return res_num;
	}
}
