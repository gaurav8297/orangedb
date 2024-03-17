#pragma once

#include <unistd.h>
#include <storage.h>
#include <x86intrin.h>
#include <macros.h>
#include <unordered_set>

using namespace std;

namespace orangedb {
    struct DistanceComputer {
        virtual void compute_distance(storage_idx_t id, float& result) = 0;
        virtual void compute_distance(storage_idx_t src, storage_idx_t dest, float& result) = 0;
        virtual void compute_distance_four_vecs(
                const storage_idx_t y0,
                const storage_idx_t y1,
                const storage_idx_t y2,
                const storage_idx_t y3,
                float& res0,
                float& res1,
                float& res2,
                float& res3) = 0;
        virtual void compute_approx_dist(storage_idx_t src, storage_idx_t dest, float& result) = 0;
        virtual storage_idx_t get_query_id() = 0;
        virtual ~DistanceComputer() = default;
    };

    struct L2DistanceComputer : public DistanceComputer {
        explicit L2DistanceComputer(const Storage* storage): storage(storage), query(nullptr), query_id(-1) {
        }

        void compute_distance(storage_idx_t id, float& result) override {
            const float* y = storage->data + (id * storage->dim);
            l2_sqr_dist(query, y, storage->dim, result);
        }

        void compute_distance(storage_idx_t src, storage_idx_t dest, float& result) override {
            const float *x = storage->data + (src * storage->dim);
            const float *y = storage->data + (dest * storage->dim);
            l2_sqr_dist(x, y, storage->dim, result);
        }

        void compute_distance_four_vecs(
                const storage_idx_t y0,
                const storage_idx_t y1,
                const storage_idx_t y2,
                const storage_idx_t y3,
                float& res0,
                float& res1,
                float& res2,
                float& res3) override {
            const float *y0_ = storage->data + (y0 * storage->dim);
            const float *y1_ = storage->data + (y1 * storage->dim);
            const float *y2_ = storage->data + (y2 * storage->dim);
            const float *y3_ = storage->data + (y3 * storage->dim);
            fvec_L2sqr_batch_4(query, y0_, y1_, y2_, y3_, storage->dim, res0, res1, res2, res3);
        }

        void compute_approx_dist(storage_idx_t src, storage_idx_t dest, float& result) override {
            const float *x = storage->data + (src * storage->dim);
            const float *y = storage->data + (dest * storage->dim);
            l1_dist(x, y, storage->dim, result);
        }

        void set_query(const float* query, const storage_idx_t query_id) {
            this->query = query;
            this->query_id = query_id;
        }

        storage_idx_t get_query_id() override {
            return query_id;
        }

    private:
#ifdef __AVX2__
        inline void l2_sqr_dist(const float* __restrict x, const float* __restrict y, size_t d, float& result) {
#define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
          tmp1 = _mm256_loadu_ps(addr1);                  \
          tmp2 = _mm256_loadu_ps(addr2);                  \
          tmp1 = _mm256_sub_ps(tmp1, tmp2);               \
          tmp1 = _mm256_mul_ps(tmp1, tmp1);               \
          dest = _mm256_add_ps(dest, tmp1);

            __m256 sum;
            __m256 l0, l1;
            __m256 r0, r1;
            size_t qty16 = d >> 4;
            size_t aligned_size = qty16 << 4;
            const float *l = x;
            const float *r = y;

            float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
            sum = _mm256_loadu_ps(unpack);

            for (unsigned i = 0; i < aligned_size; i += 32, l += 32, r += 32) {
                AVX_L2SQR(l, r, sum, l0, r0);
                AVX_L2SQR(l + 8, r + 8, sum, l1, r1);
                AVX_L2SQR(l + 16, r + 16, sum, l1, r1);
                AVX_L2SQR(l + 24, r + 24, sum, l1, r1);
            }
            _mm256_storeu_ps(unpack, sum);
            result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];
            for (unsigned i = aligned_size; i < d; ++i, ++l, ++r) {
                float diff = *l - *r;
                result += diff * diff;
            }
        }

        inline void l1_dist(const float* __restrict x, const float* __restrict y, size_t d, float& result) {
#define AVX_L1(addr1, addr2, dest, tmp1, tmp2, sign_bit) \
          tmp1 = _mm256_loadu_ps(addr1);                 \
          tmp2 = _mm256_loadu_ps(addr2);                 \
          tmp1 = _mm256_sub_ps(tmp1, tmp2);              \
          tmp1 = _mm256_andnot_ps(sign_bit, tmp1);       \
          dest = _mm256_add_ps(dest, tmp1);

            __m256 sum;
            __m256 l0, l1;
            __m256 r0, r1;
            size_t qty16 = d >> 4;
            size_t aligned_size = qty16 << 4;
            const float *l = x;
            const float *r = y;

            float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
            sum = _mm256_loadu_ps(unpack);
            __m256 sign_bit = _mm256_set1_ps(-0.0f);
            for (unsigned i = 0; i < aligned_size; i += 16, l += 16, r += 16) {
                AVX_L1(l, r, sum, l0, r0, sign_bit);
                AVX_L1(l + 8, r + 8, sum, l1, r1, sign_bit);
            }
            _mm256_storeu_ps(unpack, sum);
            result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];
        }
#else
        PRAGMA_IMPRECISE_FUNCTION_BEGIN
        inline void l2_sqr_dist(const float* __restrict x, const float* __restrict y, size_t d, float& result) {
            float res = 0;
            PRAGMA_IMPRECISE_LOOP
            for (size_t i = 0; i < d; i++) {
                float tmp = x[i] - y[i];
                res += tmp * tmp;
            }
            result = res;
        }
        PRAGMA_IMPRECISE_FUNCTION_END

        inline void l1_dist(const float* __restrict x, const float* __restrict y, size_t d, float& result) {
            // If AVX2 is not available, the cost of l1 distance is higher than l2 distance
            l2_sqr_dist(x, y, d, result);
        }
#endif
        PRAGMA_IMPRECISE_FUNCTION_BEGIN
        inline void fvec_L2sqr_batch_4(
                const float* __restrict x,
                const float* __restrict y0,
                const float* __restrict y1,
                const float* __restrict y2,
                const float* __restrict y3,
                const size_t d,
                float& dis0,
                float& dis1,
                float& dis2,
                float& dis3) {
            float d0 = 0;
            float d1 = 0;
            float d2 = 0;
            float d3 = 0;
            PRAGMA_IMPRECISE_LOOP
            for (size_t i = 0; i < d; ++i) {
                const float q0 = x[i] - y0[i];
                const float q1 = x[i] - y1[i];
                const float q2 = x[i] - y2[i];
                const float q3 = x[i] - y3[i];
                d0 += q0 * q0;
                d1 += q1 * q1;
                d2 += q2 * q2;
                d3 += q3 * q3;
            }

            dis0 = d0;
            dis1 = d1;
            dis2 = d2;
            dis3 = d3;
        }
        PRAGMA_IMPRECISE_FUNCTION_END

// TODO: add more distance functions with simd support
    private:
        const float* query;
        int64_t query_id;
        const Storage* storage;
    };
} // namespace orangedb
