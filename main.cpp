#include <cuda_runtime.h>
#include <iostream>

#include "scan.h"
#include "test.h"
#include "config.h"

using namespace std;

void InitConfigByCudaDeviceProp() {
	int device_id;
	cudaGetDevice(&device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	using namespace cui;
	Config::block_size = prop.maxThreadsPerBlock;
	Config::block_size_2_pow = 1;
	while (Config::block_size_2_pow * 2 <= Config::block_size)
		Config::block_size_2_pow *= 2;
}

int main() {
	InitConfigByCudaDeviceProp();

	cui::TestInplaceExclusiveScan();
	cui::TestInplaceInclusiveScan();
	cui::TestInplaceStreamCompression();
	cui::TestMarchingCube();
	return 0;
}
