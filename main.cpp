#include <cuda_runtime.h>
#include <iostream>

#include "scan.h"
#include "test.h"
#include "config.h"

//
// 程序开头自动检测maxThreadPerBlock和maxGridSize来计算最优的线程数分配方案
//
// MC中：
//	分支索引化来减少控制分歧。分支索引表放在常数存储器中
//	sdf和positions的shared_memory使用 以及 利用dram burst实现接合访问
// 
// Scan:
//	因为一定是都在一个block里的，所以可以把数据全部都先加载进shared_memory里
//	因为一个block里的线程会访问连续的数组位置，所以实现了接合访问
// 
// 流压缩：
//	MapIfLeftDiff可以利用一下shared_memory来减少stencil的访问（或者直接利用L2缓存）
//
//

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

	cout << "Marching Cube: " << cui::TestMarchingCube() << "ms" << endl;

	return 0;
}
