#pragma once

#include <chrono>
#include <functional>

namespace cui {
	class Timer {
	public:
		static double MeasureTime(std::function<void(void)> f, unsigned int loop_num = 500) {
			using namespace std::chrono;

			auto start = high_resolution_clock::now();
			for (int i = 0; i < loop_num; i++)
				f();
			auto end = high_resolution_clock::now();
			auto duration = duration_cast<milliseconds>(end - start);
			return duration.count() / double(loop_num);
		}
	};
}
