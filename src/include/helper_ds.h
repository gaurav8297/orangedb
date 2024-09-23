#pragma once

#include <unistd.h>
#include <vector>
#include <common.h>
#include <omp.h>
#include <atomic>
#include "storage.h"
#include "thread"
#include <unordered_set>
#include "queue"
#include "distance.h"

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
        explicit NodeDistFarther(vector_idx_t id = INVALID_VECTOR_ID, double dist = -1)
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

        int size();

    private:
        int getRandQueueIndex() const;

    public:
        std::vector<std::unique_ptr<BinaryHeap<T>>> queues;
        std::vector<std::atomic_int> queueSizes;
        const int maxSize;
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
            if (visited[id].load(std::memory_order_relaxed) == visited_id) {
                return false;
            }
            // Reduce the number of CAS operations
            uint8_t expected = 0;
            return visited[id].compare_exchange_weak(expected, visited_id);
        }

        inline void printCount() {
            int count = 0;
            for (auto &v: visited) {
                if (v.load() == visited_id) {
                    count++;
                }
            }
            printf("Visited: %d\n", count);
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

    // Task scheduler to do node expansion and distance computation in parallel
    class PocTaskScheduler {
    public:
        const int numThreads;

        explicit PocTaskScheduler(
                int numThreads,
                AtomicVisitedTable *visited,
                const int nodeExpansionPerNode,
                Storage *storage,
                const int efSearch)
                : numThreads(numThreads), stop(false), tasksInProgress(0), visited(visited), efSearch(efSearch),
                  nodeExpansionPerNode(nodeExpansionPerNode), storage(storage), resultPq(nullptr), candidates(nullptr),
                  nextFrontier(nullptr), dc(nullptr) {
            // Initialize atomic values
            nodeExpansion.store(0);
            workIdx.store(-1);
            compDist.store(false);
            startPerThread = std::vector<std::atomic_uint32_t>(numThreads);
            endPerThread = std::vector<std::atomic_uint32_t>(numThreads);

            // Initialize the thread pool
            for (size_t i = 0; i < numThreads; ++i) {
                workers.emplace_back(&PocTaskScheduler::workerThread, this, i);
            }
        }

        int parallelize_and_wait(int start, int end, bool computeDist) {
            this->compDist.store(computeDist);
            tasksInProgress.store(numThreads);  // Set total tasks in progress
            nodeExpansion.store(0);

            // Divide work into chunks for each thread
            int range = (end - start) / numThreads;
            for (int i = 0; i < numThreads; ++i) {
                startPerThread[i] = start + i * range;
                endPerThread[i] = (i == numThreads - 1) ? end : start + (i + 1) * range;
            }

            // Initialize work index for worker threads
            workIdx.store(numThreads - 1);

            // Wait for all threads to complete
            wait_for_completion();

            return nodeExpansion.load();
        }

        inline void wait_for_completion() {
            while (tasksInProgress > 0) {
                std::this_thread::yield();  // Yield to allow other threads to work
            }
        }

        ~PocTaskScheduler() {
            stop.store(true);
            for (std::thread &worker: workers) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
        }

    private:
        // Worker thread function
        void workerThread(int threadId) {
            while (!stop.load()) {
                if (workIdx < 0) {
                    std::this_thread::yield();
                    continue;
                }
                int idx = workIdx.fetch_sub(1);
                if (idx < 0) {
                    workIdx.store(-1);
                    continue;
                }

                int start = startPerThread[idx].load();
                int end = endPerThread[idx].load();
                bool isComputeDist = compDist.load();

                if (isComputeDist) {
                    for (int i = start; i < end; ++i) {
                        auto &neighbor = nextFrontier[i];
                        dc->computeDistance(neighbor.id, &neighbor.dist);

                        // Reduce the number of push operations
                        if (resultPq->size() < efSearch || neighbor.dist < resultPq->top()->dist) {
                            resultPq->push(NodeDistFarther(neighbor.id, neighbor.dist));
                        }
                    }
                    // Sort the next frontier after processing
                    std::sort(nextFrontier + start, nextFrontier + end);
                } else {
                    int totalExpansion = 0;
                    for (int i = start; i < end; ++i) {
                        if (candidates[i].isInvalid()) {
                            continue;
                        }
                        int s = findNextKNeighbours(candidates[i].id, nextFrontier + (i * nodeExpansionPerNode),
                                                    visited, nodeExpansionPerNode, 128);
                        totalExpansion += s;
                    }
                    nodeExpansion.fetch_add(totalExpansion);
                }

                // Decrement tasksInProgress
                tasksInProgress.fetch_sub(1);
            }
        }

        int findNextKNeighbours(
                vector_idx_t entrypoint,
                NodeDistCloser *nbrs,
                AtomicVisitedTable *visited,
                int maxK,
                int maxNeighboursCheck) {
            auto neighbors = storage->get_neighbors(0);
            std::queue<vector_idx_t> cd;
            cd.push(entrypoint);
            std::unordered_set<vector_idx_t> visitedSet;

            int neighboursChecked = 0;
            int m = 0;

            while (neighboursChecked <= maxNeighboursCheck && !cd.empty()) {
                auto candidate = cd.front();
                cd.pop();

                size_t begin, end;
                if (visitedSet.contains(candidate)) {
                    continue;
                }
                visitedSet.insert(candidate);
                storage->get_neighbors_offsets(candidate, 0, begin, end);

                neighboursChecked++;

                for (size_t i = begin; i < end; ++i) {
                    auto neighbor = neighbors[i];
                    if (neighbor == INVALID_VECTOR_ID) {
                        break;
                    }

                    if (visited->getAndSet(neighbor)) {
                        nbrs[m] = NodeDistCloser(neighbor, std::numeric_limits<double>::max());
                        m++;
                        if (m >= maxK) {
                            return m;
                        }
                    }
                    cd.push(neighbor);
                }
            }
            return m;
        }

    public:
        ParallelMultiQueue<NodeDistFarther> *resultPq;
        NodeDistCloser *candidates;
        NodeDistCloser *nextFrontier;
        DistanceComputer *dc;

    private:
        const int efSearch;
        AtomicVisitedTable *visited;
        const int nodeExpansionPerNode;
        Storage *storage;

        std::vector<std::thread> workers;
        std::atomic_uint32_t nodeExpansion;
        std::vector<std::atomic_uint32_t> startPerThread;
        std::vector<std::atomic_uint32_t> endPerThread;
        std::atomic_int8_t workIdx;
        std::atomic_bool compDist;
        std::atomic_uint8_t tasksInProgress;
        std::atomic_bool stop;
    };
} // namespace orangedb
