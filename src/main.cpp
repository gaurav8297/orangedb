#include <iostream>
#include "aux_ds.h"
using namespace orangedb;

void benchmark_fast_l2_distance() {
    std::cout << "benchmark_fast_l2_distance" << std::endl;

}


int main() {
    MaxHeap maxHeap(5);
    maxHeap.push(1, 15.0);
    maxHeap.push(2, 10.0);
    maxHeap.push(3, 20.0);
    maxHeap.push(4, 5.0);
    maxHeap.push(5, 25.0);
    maxHeap.push(6, 22.0);
    return 0;
}
