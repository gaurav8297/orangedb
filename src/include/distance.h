#pragma once

#include <unistd.h>
#include <storage.h>

#ifdef __AVX2__
#include <x86intrin.h>
#endif

#include <macros.h>
#include <unordered_set>
#include <prefetch.h>
#include <scalar_quantizer.h>
#include <assert.h>
#include <memory>

using namespace std;

namespace orangedb {
    struct DistanceComputer {
        virtual void computeDistance(storage_idx_t id, float &result) = 0;

        virtual void computeDistance(storage_idx_t src, storage_idx_t dest, float &result) = 0;

        virtual void computeDistanceFourVecs(
                storage_idx_t y0,
                storage_idx_t y1,
                storage_idx_t y2,
                storage_idx_t y3,
                float &res0,
                float &res1,
                float &res2,
                float &res3) = 0;

        virtual void setQuery(const float *query) = 0;

        virtual std::unique_ptr<DistanceComputer> clone() = 0;

        virtual ~DistanceComputer() = default;
    };

    struct L2DistanceComputer : public DistanceComputer {
        explicit L2DistanceComputer(const float *data, int dim, int n) : data(data), dim(dim), n(n), query(nullptr) {}

        inline void computeDistance(storage_idx_t id, float &result) override {
            assert(id < n);
            const float *y = data + (id * dim);
            l2_sqr_dist(query, y, dim, result);
        }

        inline void computeDistance(storage_idx_t src, storage_idx_t dest, float &result) override {
            assert(src < n && dest < n);
            const float *x = data + (src * dim);
            const float *y = data + (dest * dim);
            l2_sqr_dist(x, y, dim, result);
        }

        inline void computeDistanceFourVecs(
                const storage_idx_t y0,
                const storage_idx_t y1,
                const storage_idx_t y2,
                const storage_idx_t y3,
                float &res0,
                float &res1,
                float &res2,
                float &res3) override {
            assert(y0 < n && y1 < n && y2 < n && y3 < n);
            const float *y0_ = data + (y0 * dim);
            const float *y1_ = data + (y1 * dim);
            const float *y2_ = data + (y2 * dim);
            const float *y3_ = data + (y3 * dim);
            fvec_L2sqr_batch_4(query, y0_, y1_, y2_, y3_, dim, res0, res1, res2, res3);
        }

        inline void setQuery(const float *query) override {
            this->query = query;
        }

        inline std::unique_ptr<DistanceComputer> clone() override {
            return std::make_unique<L2DistanceComputer>(data, dim, n);
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

            int j = 0;
            for (unsigned i = 0; i < aligned_size; i += 16, l += 16, r += 16) {
//                if (i + 16 < aligned_size) {
//                    prefetch_NTA(l + 16 * (j + 1));
//                    prefetch_NTA(r + 16 * (j + 1));
//                    prefetch_NTA(l + 24 * (j + 1));
//                    prefetch_NTA(r + 24 * (j + 1));
//                }
                AVX_L2SQR(l, r, sum, l0, r0);
                AVX_L2SQR(l + 8, r + 8, sum, l1, r1);
                j++;
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
        inline void l2_sqr_dist(const float *__restrict x, const float *__restrict y, size_t d, float &result) {
            float res = 0;
            PRAGMA_IMPRECISE_LOOP
            for (size_t i = 0; i < d; i++) {
                float tmp = x[i] - y[i];
                res += tmp * tmp;
            }
            result = res;
        }

        PRAGMA_IMPRECISE_FUNCTION_END

        inline void l1_dist(const float *__restrict x, const float *__restrict y, size_t d, float &result) {
            // If AVX2 is not available, the cost of l1 distance is higher than l2 distance
            l2_sqr_dist(x, y, d, result);
        }

#endif
        PRAGMA_IMPRECISE_FUNCTION_BEGIN
        inline void fvec_L2sqr_batch_4(
                const float *__restrict x,
                const float *__restrict y0,
                const float *__restrict y1,
                const float *__restrict y2,
                const float *__restrict y3,
                const size_t d,
                float &dis0,
                float &dis1,
                float &dis2,
                float &dis3) {
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

    private:
        const float *data;
        int dim;
        int n;

        const float *query;
    };

    struct SQDistanceComputer : public DistanceComputer {
        explicit SQDistanceComputer(const uint8_t *data, int dim, int n, const float *vmin, const float *vdiff)
                : data(data), dim(dim), n(n), vmin(vmin), vdiff(vdiff),
                  query(nullptr) {};

        void computeDistance(storage_idx_t id, float &result) override {
            const uint8_t *y = data + (id * dim);
            l2_sqr_dist(y, dim, result);
        }

        void computeDistance(storage_idx_t src, storage_idx_t dest, float &result) override {
            const uint8_t *x = data + (src * dim);
            const uint8_t *y = data + (dest * dim);
            l2_sqr_dist_2(x, y, dim, result);
        }

        void computeDistanceFourVecs(
                const storage_idx_t y0,
                const storage_idx_t y1,
                const storage_idx_t y2,
                const storage_idx_t y3,
                float &res0,
                float &res1,
                float &res2,
                float &res3) override {
            // throw error
            std::runtime_error("Not implemented");
        }

        void setQuery(const float *query) override {
            this->query = query;
        }

        std::unique_ptr<DistanceComputer> clone() override {
            return std::make_unique<SQDistanceComputer>(data, dim, n, vmin, vdiff);
        }

#ifdef __AVX2__
        inline void l2_sqr_dist(const uint8_t* codes, size_t d, float& result) {
#define AVX_L2SQR_1(addr1, vec2, dest, tmp1) \
          tmp1 = _mm256_loadu_ps(addr1);                  \
          tmp1 = _mm256_sub_ps(tmp1, vec2);               \
          dest = _mm256_fmadd_ps(tmp1, tmp1, dest);
            __m256 sum;
            __m256 l0, l1;
            size_t qty16 = d >> 4;
            size_t aligned_size = qty16 << 4;
            const float *l = query;

            float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
            sum = _mm256_loadu_ps(unpack);

            for (unsigned i = 0; i < aligned_size; i += 16, l += 16) {
                auto yvec = reconstruct_8_components(codes, i, vmin, vdiff);
                AVX_L2SQR_1(l, yvec, sum, l0);
                auto yvec_next_8 = reconstruct_8_components(codes, i + 8, vmin, vdiff);
                AVX_L2SQR_1(l + 8, yvec_next_8, sum, l1);
            }
            _mm256_storeu_ps(unpack, sum);
            result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];
//            for (unsigned i = aligned_size; i < d; ++i, ++l) {
//                float diff = *l - *r;
//                result += diff * diff;
//            }
        }

        inline void l2_sqr_dist_2(const uint8_t* xcodes, const uint8_t* ycodes, size_t d, float& result) {
#define AVX_L2SQR_2(vec1, vec2, dest, tmp1) \
          tmp1 = _mm256_sub_ps(vec1, vec2);               \
          dest = _mm256_fmadd_ps(tmp1, tmp1, dest);
            __m256 sum;
            __m256 l0, l1;
            size_t qty16 = d >> 4;
            size_t aligned_size = qty16 << 4;
            const float *l = query;

            float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
            sum = _mm256_loadu_ps(unpack);

            for (unsigned i = 0; i < aligned_size; i += 16, l += 16) {
                auto xvec = reconstruct_8_components(xcodes, i, vmin, vdiff);
                auto yvec = reconstruct_8_components(ycodes, i, vmin, vdiff);
                AVX_L2SQR_2(xvec, yvec, sum, l0);
                auto xvec_next_8 = reconstruct_8_components(xcodes, i + 8, vmin, vdiff);
                auto yvec_next_8 = reconstruct_8_components(ycodes, i + 8, vmin, vdiff);
                AVX_L2SQR_2(xvec_next_8, yvec_next_8, sum, l1);
            }
            _mm256_storeu_ps(unpack, sum);
            result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];
//            for (unsigned i = aligned_size; i < d; ++i, ++l) {
//                float diff = *l - *r;
//                result += diff * diff;
//            }
        }

        static inline __m256 reconstruct_8_components(const uint8_t *code, int i, const float *vmin, const float *vdiff) {
            const uint64_t c8 = *(uint64_t *) (code + i);
            // load 8, 8 bit integer into __m128i
            const __m128i i8 = _mm_set1_epi64x(c8);
            // Convert 8 bit to 32 bit int
            const __m256i i32 = _mm256_cvtepu8_epi32(i8);
            // Convert 32 bit to floating point number
            const __m256 f8 = _mm256_cvtepi32_ps(i32);
            // constants
            const __m256 half_one_255 = _mm256_set1_ps(0.5f / 255.f);
            const __m256 one_255 = _mm256_set1_ps(1.f / 255.f);
            // calculate the scaled value between [0, 1]
            const __m256 scaled_val = _mm256_fmadd_ps(f8, one_255, half_one_255);
            // vmin + scaled_val * vdiff -> return actual float value
            return _mm256_fmadd_ps(
                    scaled_val,
                    _mm256_loadu_ps(vdiff + i),
                    _mm256_loadu_ps(vmin + i));
        }
#else
        // Auto vectorization doesn't work for this function. Maybe simplify the function to make it work
        PRAGMA_IMPRECISE_FUNCTION_BEGIN
        inline void l2_sqr_dist(const uint8_t *codes, size_t d, float &result) {
            float res = 0;
            PRAGMA_IMPRECISE_LOOP
            for (size_t i = 0; i < d; i++) {
                float decoded_val = vmin[i] + (codes[i] + 0.5f) / 255.0f * vdiff[i];
                float diff = query[i] - decoded_val;
                res += diff * diff;
            }
            result = res;
        }

        PRAGMA_IMPRECISE_FUNCTION_END

        // Auto vectorization doesn't work for this function. Maybe simplify the function to make it work
        PRAGMA_IMPRECISE_FUNCTION_BEGIN
        inline void l2_sqr_dist_2(const uint8_t *xcodes, const uint8_t *ycodes, size_t d, float &result) {
            float res = 0;
            PRAGMA_IMPRECISE_LOOP
            for (size_t i = 0; i < d; i++) {
                float decoded_val_x = vmin[i] + (xcodes[i] + 0.5f) / 255.0f * vdiff[i];
                float decoded_val_y = vmin[i] + (ycodes[i] + 0.5f) / 255.0f * vmin[i];
                float diff = decoded_val_x - decoded_val_y;
                res += diff * diff;
            }
            result = res;
        }

        PRAGMA_IMPRECISE_FUNCTION_END
#endif

    private:
        const uint8_t *data;
        int dim;
        int n;
        const float *vmin;
        const float *vdiff;

        const float *query;
    };
} // namespace orangedb
