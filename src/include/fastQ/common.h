#pragma once

#include <functional>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <random>
#include <iostream>
#include <format>
#include <unordered_set>

namespace fastq {
    namespace common {
        // Types
        typedef uint64_t vector_id_t;
        constexpr vector_id_t INVALID_VECTOR_IDX = UINT64_MAX;

        typedef uint32_t centroid_id_t;
        constexpr centroid_id_t INVALID_CENTROID_IDX = UINT32_MAX;

        // Verify function
        [[noreturn]] inline void fail_verify(
                const char *condition_name, const char *file, int linenr, const char *comment) {
            throw std::invalid_argument("Invalid argument");
        }

#define verify(condition, comment)                                                                  \
    static_cast<bool>(condition) ?                                                                  \
        void(0) :                                                                                   \
        fail_verify(#condition, __FILE__, __LINE__, static_cast<const char*>(comment));

        // Task scheduler
        struct Task {
            int id;
            int start;
            int end;
            std::function<void(int, int, int)> job;
        };

        class TaskScheduler {
        public:
            explicit TaskScheduler() : TaskScheduler(std::thread::hardware_concurrency() * 2) {};

            explicit TaskScheduler(int numThreads) : stop(false), tasksInProgress(0) {
                // Initialize the thread pool
                for (size_t i = 0; i < numThreads; ++i) {
                    workers.emplace_back(&TaskScheduler::workerThread, this, i);
                }
            };

            void parallelize_and_wait(int start, int end, int batchSize, std::function<void(int, int, int)> batchJob) {
                int taskId = parallelize(start, end, batchSize, batchJob);
                wait_for_completion(taskId);
            }

            int parallelize(int start, int end, int batchSize, std::function<void(int, int, int)> batchJob) {
                int taskId = 0;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    taskId = nextTaskId++;
                    int taskCount = 0;
                    for (int i = start; i < end; i += batchSize) {
                        tasks.push({taskId, i, std::min(i + batchSize, end), batchJob});
                        taskCount++;
                    }
                    tasksInProgress[taskId] = taskCount;
                }
                condition.notify_all();
                return taskId;
            }

            void wait_for_completion(int taskId) {
                std::unique_lock<std::mutex> lock(mutex);
                completionCondition.wait(lock, [&] { return tasksInProgress[taskId] == 0; });
            }

            ~TaskScheduler() {
                stop.store(true);
                condition.notify_all();
                for (std::thread &worker: workers) {
                    worker.join();
                }
            }

        private:
            void workerThread(int threadId) {
                while (!stop) {
                    Task task;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        condition.wait(lock, [this] { return !tasks.empty() || stop; });
                        if (stop) {
                            return;
                        }
                        // Todo: Maybe replace with a lock-free queue
                        task = tasks.front();
                        tasks.pop();
                    }
                    task.job(threadId, task.start, task.end);
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        if (--tasksInProgress[task.id] == 0) {
                            completionCondition.notify_all();
                        }
                    }
                }
            }

        private:
            std::vector<std::thread> workers;
            std::mutex mutex;
            std::queue<Task> tasks;
            std::condition_variable condition;
            std::condition_variable completionCondition;
            std::atomic<bool> stop;
            int nextTaskId = 0;
            std::unordered_map<int, int> tasksInProgress;
        };

        // Random generator
        struct RandomGenerator {
            explicit RandomGenerator(int seed) : mt(seed) {};

            inline float rand_float() {
                return mt() / float(mt.max());
            }

            inline uint64_t rand_int(uint64_t max) {
                return mt() % max;
            }

            inline void random_perm(uint64_t n, uint64_t *perm, uint64_t nPerm) {
                verify(nPerm <= n, "Number of permutations should be less than the number of elements");
                std::unordered_map<uint64_t, uint64_t> m;
                std::unordered_set<uint64_t> used_indices;
                for (int i = 0; i < nPerm; i++) {
                    uint64_t i2;
                    do {
                        i2 = next_random(i, n, m);
                    } while (used_indices.contains(i2) || n - i == 1);
                    perm[i] = i2;
                    used_indices.insert(i2);
                }
            }

        private:
            inline uint64_t next_random(uint64_t i, uint64_t n, std::unordered_map<uint64_t, uint64_t> &m) {
                uint64_t i2 = i + rand_int(n - i);
                uint64_t result;
                if (m.contains(i2)) {
                    result = m[i2];
                } else {
                    result = i2;
                }
                m[i2] = i;
                return result;
            }

        private:
            std::mt19937 mt;
        };

        // Vector hash & equal functions
        struct VectorHash {
            std::size_t operator()(const std::vector<float> &vec) const {
                std::size_t seed = 0;
                for (float num: vec) {
                    // Hash the individual float element
                    std::size_t hash = std::hash<float>{}(num);
                    // Combine the hash using XOR and bit shifting
                    seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                }
                return seed;
            }
        };

        struct VectorEqual {
            bool operator()(const std::vector<float> &lhs, const std::vector<float> &rhs) const {
                return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
            }
        };
    }
}
