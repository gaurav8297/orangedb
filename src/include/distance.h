#pragma once

#include <unistd.h>
#include <unordered_set>
#include <memory>
#include <simsimd/simsimd.h>
#include <common.h>

using namespace std;

namespace orangedb {
    struct DistanceComputer {
        virtual void computeDistance(vector_idx_t id, double *result) = 0;

        virtual void computeDistance(vector_idx_t src, vector_idx_t dest, double *result) = 0;

        virtual void computeDistance(vector_idx_t src, vector_idx_t dest, int dim, double *result) = 0;

        virtual void batchComputeDistances(vector_idx_t *ids, double *results, int size) = 0;

        virtual void setQuery(const float *query) = 0;

        virtual std::unique_ptr<DistanceComputer> clone() = 0;

        virtual ~DistanceComputer() = default;
    };

    struct L2DistanceComputer : public DistanceComputer {
        explicit L2DistanceComputer(const float *data, int dim, int n) : data(data), dim(dim), n(n), query(nullptr) {}

        inline void computeDistance(vector_idx_t id, double *result) override {
            CHECK_ARGUMENT(id < n, fmt::format("Index out of bounds {} < {}", id, n).data());
            const float *y = data + (id * dim);
            simsimd_l2sq_f32(query, y, dim, result);
        }

        inline void computeDistance(vector_idx_t src, vector_idx_t dest, double *result) override {
            CHECK_ARGUMENT(src < n && dest < n, "Index out of bounds");
            const float *x = data + (src * dim);
            const float *y = data + (dest * dim);
            simsimd_l2sq_f32(x, y, dim, result);
        }

        inline void computeDistance(vector_idx_t src, vector_idx_t dest, int dim, double *result) override {
            CHECK_ARGUMENT(src < n && dest < n, "Index out of bounds");
            CHECK_ARGUMENT(dim > 0 && dim <= this->dim, "Invalid dimension");
            const float *x = data + (src * dim);
            const float *y = data + (dest * dim);
            simsimd_l2sq_f32(x, y, dim, result);
        }

        inline void batchComputeDistances(vector_idx_t *ids, double *results, int size) override {
            for (int i = 0; i < size; i++) {
                computeDistance(ids[i], &results[i]);
            }
        }

        inline void setQuery(const float *q) override {
            this->query = q;
        }

        inline std::unique_ptr<DistanceComputer> clone() override {
            return std::make_unique<L2DistanceComputer>(data, dim, n);
        }

    private:
        const float *data;
        int dim;
        int n;

        const float *query;
    };

    struct CosineDistanceComputer : public DistanceComputer {
        explicit CosineDistanceComputer(const float *data, int dim, int n) : data(data), dim(dim), n(n),
                                                                             query(nullptr) {}

        inline void computeDistance(vector_idx_t id, double *result) override {
            CHECK_ARGUMENT(id < n, fmt::format("Index out of bounds {} < {}", id, n).data());
            const float *y = data + (id * dim);
            simsimd_cos_f32(query, y, dim, result);
        }

        inline void computeDistance(vector_idx_t src, vector_idx_t dest, double *result) override {
            CHECK_ARGUMENT(src < n && dest < n, "Index out of bounds");
            const float *x = data + (src * dim);
            const float *y = data + (dest * dim);
            simsimd_cos_f32(x, y, dim, result);
        }

        inline void computeDistance(vector_idx_t src, vector_idx_t dest, int dim, double *result) override {
            CHECK_ARGUMENT(src < n && dest < n, "Index out of bounds");
            CHECK_ARGUMENT(dim > 0 && dim <= this->dim, "Invalid dimension");
            const float *x = data + (src * dim);
            const float *y = data + (dest * dim);
            simsimd_cos_f32(x, y, dim, result);
        }

        inline void batchComputeDistances(vector_idx_t *ids, double *results, int size) override {
            for (int i = 0; i < size; i++) {
                computeDistance(ids[i], &results[i]);
            }
        }

        inline void setQuery(const float *q) override {
            this->query = q;
        }

        inline std::unique_ptr<DistanceComputer> clone() override {
            return std::make_unique<CosineDistanceComputer>(data, dim, n);
        }

    private:
        const float *data;
        int dim;
        int n;

        const float *query;
    };
} // namespace orangedb
