#include <iostream>

#include "scan.h"
#include "test.h"

using namespace std;

int main() {
	cui::TestInplaceExclusiveScan();
	cui::TestInplaceInclusiveScan();
	cui::TestInplaceStreamCompression();
	cui::TestMarchingCube();
	return 0;
}
