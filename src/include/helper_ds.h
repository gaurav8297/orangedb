#pragma once

#include <unistd.h>
#include <vector>
#include <common.h>
#include <omp.h>
#include <atomic>

using namespace std;

namespace orangedb {
    struct NodeDistCloser {
        explicit NodeDistCloser(vector_idx_t id = INVALID_VECTOR_ID, double dist = std::numeric_limits<double>::max())
                : id(id), dist(dist) {}

        vector_idx_t id;
        double dist;

        bool operator<(const NodeDistCloser &other) const {
            return dist < other.dist;
        }

        bool operator>(const NodeDistCloser &other) const {
            return dist > other.dist;
        }

        bool operator>= (const NodeDistCloser &other) const {
            return dist >= other.dist;
        }

        bool operator<= (const NodeDistCloser &other) const {
            return dist <= other.dist;
        }

        bool operator==(const NodeDistCloser &other) const {
            return id == other.id && dist == other.dist;
        }

        inline bool isInvalid() const {
            return id == INVALID_VECTOR_ID;
        }
    };

    struct NodeDistFarther {
        explicit NodeDistFarther(vector_idx_t id = INVALID_VECTOR_ID, double dist = 0.0)
                : id(id), dist(dist) {}

        vector_idx_t id;
        double dist;

        bool operator<(const NodeDistFarther &other) const {
            return dist > other.dist;
        }

        bool operator>(const NodeDistFarther &other) const {
            return dist < other.dist;
        }

        bool operator>= (const NodeDistFarther &other) const {
            return dist <= other.dist;
        }

        bool operator<= (const NodeDistFarther &other) const {
            return dist >= other.dist;
        }

        bool operator==(const NodeDistFarther &other) const {
            return id == other.id && dist == other.dist;
        }

        inline bool isInvalid() const {
            return id == INVALID_VECTOR_ID;
        }
    };

    template<typename T>
    class BinaryHeap {
    public:
        explicit BinaryHeap(int capacity);

        void push(T val);

        T popMin();

        inline const int size() const {
            return actual_size;
        };

        inline const T *top() const {
            return &nodes[1];
        }

        inline void lock() {
            omp_set_lock(&mtx);
        }

        inline void unlock() {
            omp_unset_lock(&mtx);
        }

        inline const T *getMinElement() {
            return minElement.load(std::memory_order_relaxed);
        }

        ~BinaryHeap() {
            omp_destroy_lock(&mtx);
        }

    private:
        void pushToHeap(T val);

        void popMinFromHeap();

        void popMaxFromHeap();

    private:
        int capacity;
        int actual_size;
        std::vector<T> nodes;
        omp_lock_t mtx; // Spinlock

        // Redundantly store the min element separately
        std::atomic<T *> minElement;
    };

    // Based on this paper: https://arxiv.org/abs/1411.1209
    template<typename T>
    class ParallelMultiQueue {
    public:
        explicit ParallelMultiQueue(int num_queues, int reserve_size);

        void push(T val);

        T popMin();

        const T *top();

    private:
        int getRandQueueIndex() const;

    private:
        std::vector<std::unique_ptr<BinaryHeap<T>>> queues;
    };

    // TODO: Use bitset instead of vector<uint8_t>
    class VisitedTable {
    public:
        explicit VisitedTable(size_t size) : visited(size, 0), visited_id(1) {};

        inline void set(vector_idx_t id) {
            visited[id] = visited_id;
        }

        inline bool get(vector_idx_t id) {
            return visited[id] == visited_id;
        }

        inline void reset() {
            visited_id++;
            if (visited_id == 250) {
                std::fill(visited.begin(), visited.end(), 0);
                visited_id = 1;
            }
        }

        inline uint8_t *data() {
            return visited.data();
        }

    private:
        std::vector<uint8_t> visited;
        uint8_t visited_id;
    };

    class AtomicVisitedTable {
    public:
        explicit AtomicVisitedTable(size_t size) : visited(size), visited_id(1) {
            for (auto &v: visited) {
                v.store(0);
            }
        };

        inline void set(vector_idx_t id) {
            visited[id].store(visited_id);
        }

        inline bool get(vector_idx_t id) {
            return visited[id].load() == visited_id;
        }

        bool getAndSet(vector_idx_t id) {
            uint8_t expected = 0;
            return visited[id].compare_exchange_weak(expected, visited_id);
        }

//        inline void reset() {
//            visited_id++;
//            if (visited_id == 250) {
//                for (auto &v: visited) {
//                    v.store(0);
//                }
//                visited_id = 1;
//            }
//        }

        inline std::vector<std::atomic<uint8_t>> &data() {
            return visited;
        }

    private:
        std::vector<std::atomic<uint8_t>> visited;
        const uint8_t visited_id;
    };
} // namespace orangedb
