.PHONY: build debug

NUM_THREADS ?= 1

export CMAKE_BUILD_PARALLEL_LEVEL=$(NUM_THREADS)

build:
	cmake -B build/release $(CMAKE_FLAGS) -DCMAKE_BUILD_TYPE=Release $1 .
	cmake --build build/release --config Release

debug:
	cmake -B build/debug $(CMAKE_FLAGS) -DENABLE_ADDRESS_SANITIZER=1 -DCMAKE_BUILD_TYPE=Debug $1 .
	cmake --build build/debug --config Debug
