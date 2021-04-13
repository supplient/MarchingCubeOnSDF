#pragma once

#include <cuda_runtime_api.h>


namespace cui {
	template<typename CheckFunc>
	class CudaTempMem {
	public:
		CudaTempMem(unsigned int N) :N(N) {
			cudaMalloc(&d_ptr, N * sizeof(CheckFunc));
		}
		~CudaTempMem() {
			cudaFree(d_ptr);
		}

		CheckFunc* Get() {
			return d_ptr;
		}

		operator CheckFunc* () {
			return Get();
		}

	private:
		unsigned int N;
		CheckFunc* d_ptr;
	};
}
