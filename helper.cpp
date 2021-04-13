
namespace cui {
	int CalBlockNum(int total, int per_block) {
		return (total - 1) / per_block + 1;
	}
}
