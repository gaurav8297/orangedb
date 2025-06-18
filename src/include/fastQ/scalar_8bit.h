#pragma once

#include <fstream>
#include <vector>
#include <simsimd/simsimd.h>
#include <fastQ/fastq.h>

// TODO:
// 1. Add support for avx2
// 2. Add support for avx512

namespace fastq {
    namespace scalar_8bit {
        // Scalar quantization for 8-bit integers
        // Encoding:
        // 1. Scale to [0, 1] => (x - vmin) / vdiff
        // 2. Scale to [0, 255] => x * 255
        // 3. Round to the nearest integer
        //
        // Decoding:
        // 1. Scale to [0, 1] => (ci + 0.5) / 255
        // 2. Scale to [vmin, vmax] => vmin + ci * vdiff
        //
        // Reason to add 0.5f is to round the value:
        // This is used for continuity correction. Basically the probability that a random variable
        // falls in a certain range is the same as the probability that the random variable falls in the
        // corresponding integer value. It gives out a more accurate result.
        //
        // Use precomputed values to calculate the distance between two encoded vectors:
        // distance = i8 * i8' * alpha^2 + i8 * beta * alpha + i8' * beta * alpha + beta^2
        // where alpha = vdiff / 3.0f and beta = 0.5f * alpha + vmin
        //
        // We can precompute alpha^2 and gamma^2 to speed up the computation. Additionally, we can precompute
        // i8 * alpha * gamma for each vector and store it as part of the compressed data. This will improve
        // distance computation speed. Therefore, we need 4 more additional bytes per vector to store these
        // precomputed values.
        //
        // Additionally, decoding can also be written as:
        // x = alpha * ci + beta
        // TODO: Think about making precomputed values optional

       // Serial encoding functions
        inline uint8_t encode_serial(const float x, const float vmin, const float vdiff,
                                     float scalar_range) {
            auto val = std::min(std::max(x - vmin, 0.0f), vdiff);
            return std::min(int((val / vdiff) * scalar_range), (int) scalar_range - 1);
        }

        inline float compute_precomputed_value_serial(const uint8_t code, const float alpha, const float beta) {
            return ((float) code) * alpha * beta;
        }

        inline void encode_serial_8bit(const float *data, uint8_t *codes, int dim, const float *vmin,
                                       const float *vdiff, const float *alpha, const float *beta, float beta_sqr) {
            float precompute_value = beta_sqr;
            for (size_t j = 0; j < dim; j++) {
                codes[j] = encode_serial(data[j], vmin[j], vdiff[j], 256.0f);
                precompute_value += compute_precomputed_value_serial(codes[j], alpha[j], beta[j]);
            }
            // Store precomputed values
            *reinterpret_cast<float *>(codes + dim) = precompute_value;
        }

        inline float decode_serial(const uint8_t code, const float alpha, const float beta) {
            return alpha * code + beta;
        }

        inline void compute_asym_l2sq_serial_8bit(const float *x, const uint8_t *y, double *result, size_t dim,
                                                  const float *alpha, const float *beta) {
            double res = 0;
            for (size_t i = 0; i < dim; i++) {
                auto code = decode_serial(y[i], alpha[i], beta[i]);
                auto xc = (x[i] - code);
                res += xc * xc;
            }
            *result = res;
        }

        inline void compute_sym_l2sq_serial_8bit(const uint8_t *x, const uint8_t *y, double *result, size_t dim,
                                                 const float *alphaSqr) {
            double res = 0;
            for (size_t i = 0; i < dim; i++) {
                int xy = x[i] - y[i];
                res += xy * xy * alphaSqr[i];
            }
            *result = res;
        }

        inline void compute_asym_ip_serial_8bit(const float *x, const uint8_t *y, double *result, size_t dim,
                                                  const float *alpha, const float *beta) {
            double xy = 0;
            for (size_t i = 0; i < dim; i++) {
                xy += x[i] * decode_serial(y[i], alpha[i], beta[i]);
            }
            *result = xy;
        }

        inline void compute_sym_ip_serial_8bit(const uint8_t *x, const uint8_t *y, double *result, size_t dim,
                                                const float *alphaSqr) {
            double xy = 0;
            for (size_t i = 0; i < dim; i++) {
                xy += x[i] * y[i] * alphaSqr[i];
            }
            // Add precomputed value (last 4 bytes)
            xy += *reinterpret_cast<const float *>(x + dim);
            xy += *reinterpret_cast<const float *>(y + dim);
            *result = xy;
        }

#if _SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("+simd")
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)

        inline uint32x4_t encode_neon_(const float *x, size_t i, const float *vmin, const float *vdiff,
                                       const float scalar_range) {
            float32x4_t scalar_range_vec = vdupq_n_f32(scalar_range);
            float32x4_t vmin_vec = vld1q_f32(vmin + i);
            float32x4_t vdiff_vec = vld1q_f32(vdiff + i);
            float32x4_t x_vec = vld1q_f32(x + i);

            // Clamp to [vmin, vmin + vdiff]
            float32x4_t val = vminq_f32(vmaxq_f32(vsubq_f32(x_vec, vmin_vec), vdupq_n_f32(0.0f)), vdiff_vec);

            // Scale to [0, 1]
            float32x4_t x_scaled = vdivq_f32(val, vdiff_vec);
            // Scale to [0, scalar_range) => min(x * scalar_range, scalar_range - 1)
            uint32x4_t ci = vminq_u32(vcvtq_u32_f32(vmulq_f32(x_scaled, scalar_range_vec)),
                                      vdupq_n_u32((uint32_t) scalar_range - 1));
            return ci;
        }

        inline uint8x8_t encode_neon(const float *x, size_t i, const float *vmin, const float *vdiff,
                                     const float scalar_range) {
            uint32x4_t ci_low = encode_neon_(x, i, vmin, vdiff, scalar_range);
            uint32x4_t ci_high = encode_neon_(x, i + 4, vmin, vdiff, scalar_range);
            return vmovn_u16(vcombine_u16(vmovn_u32(ci_low), vmovn_u32(ci_high)));
        }

        inline float32x4_t calc_precomputed_values_neon(uint8x8_t ci_vec, size_t i,
                                                        const float *alpha, const float *beta) {
            uint16x8_t ci_vec16 = vmovl_u8(ci_vec);
            float32x4_t ci_vec32_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(ci_vec16)));
            float32x4_t ci_vec32_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(ci_vec16)));

            // Calculate precomputed values i8 * alpha * beta
            float32x4_t precompute_values_low = vmulq_f32(vmulq_f32(ci_vec32_low, vld1q_f32(alpha + i)),
                                                          vld1q_f32(beta + i));
            float32x4_t precompute_values_high = vmulq_f32(vmulq_f32(ci_vec32_high, vld1q_f32(alpha + i + 4)),
                                                           vld1q_f32(beta + i + 4));

            return vaddq_f32(precompute_values_low, precompute_values_high);
        }

        inline void encode_neon_8bit(const float *data, uint8_t *codes, int dim, const float *vmin,
                                     const float *vdiff, const float *alpha, const float *beta) {
            int i = 0;
            float32x4_t precompute_values = vdupq_n_f32(0);
            for (; i + 8 <= dim; i += 8) {
                uint8x8_t ci_vec = encode_neon(data, i, vmin, vdiff, 256.0f);
                precompute_values = vaddq_f32(calc_precomputed_values_neon(ci_vec, i, alpha, beta),
                                              precompute_values);
                vst1_u8(codes + i, ci_vec);
            }
            auto precompute_value = vaddvq_f32(precompute_values);
            // Handle the remaining
            for (; i < dim; i++) {
                codes[i] = encode_serial(data[i], vmin[i], vdiff[i], 256.0f);
                precompute_value += compute_precomputed_value_serial(codes[i], alpha[i], beta[i]);
            }
            // Store precomputed values
            *reinterpret_cast<float *>(codes + dim) = precompute_value;
        }

#pragma clang attribute pop
#pragma GCC pop_options
#endif
#endif

#if _SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "fma")
#pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to = function)

        inline __m128i encode_haswell_(const float *x, size_t i, const float *vmin, const float *vdiff,
                                       const float scalar_range) {
            __m256 scalar_range_vec = _mm256_set1_ps(scalar_range);
            __m256 vmin_vec = _mm256_loadu_ps(vmin + i);
            __m256 vdiff_vec = _mm256_loadu_ps(vdiff + i);
            __m256 x_vec = _mm256_loadu_ps(x + i);

            // Clamp to [vmin, vmin + vdiff]
            __m256 val = _mm256_min_ps(_mm256_max_ps(_mm256_sub_ps(x_vec, vmin_vec), _mm256_setzero_ps()), vdiff_vec);


            // Scale to [0, 1]
            __m256 x_scaled = _mm256_div_ps(val, vdiff_vec);

            // Scale to [0, scalar_range) => x * scalar_range
            __m256i ci = _mm256_cvtps_epi32(_mm256_mul_ps(x_scaled, scalar_range_vec));

            // Pack and clamp to [0, 255]
            __m128i ci_low = _mm256_extracti128_si256(ci, 0);
            __m128i ci_high = _mm256_extracti128_si256(ci, 1);
            __m128i packed16 = _mm_packus_epi32(ci_low, ci_high);

            return packed16;
        }

        inline __m128i encode_haswell(const float *x, size_t i, const float *vmin, const float *vdiff,
                                      const float scalar_range) {
            __m128i ci_low = encode_haswell_(x, i, vmin, vdiff, scalar_range);
            __m128i ci_high = encode_haswell_(x, i + 8, vmin, vdiff, scalar_range);
            return _mm_packus_epi16(ci_low, ci_high);
        }

        inline __m256 calc_precomputed_values_haswell(__m128i ci_vec, size_t i, const float *alpha, const float *beta) {
            // Load and extend ci_vec data
            __m256 ci_vec_lower_half = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(ci_vec));
            __m256 ci_vec_upper_half = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(ci_vec, 8)));

            // Calculate precomputed values
            __m256 precompute_values_lower_half = _mm256_mul_ps(_mm256_mul_ps(
                    ci_vec_lower_half, _mm256_loadu_ps(alpha + i)), _mm256_loadu_ps(beta + i));
            __m256 precompute_values_upper_half = _mm256_mul_ps(_mm256_mul_ps(
                    ci_vec_upper_half, _mm256_loadu_ps(alpha + i + 8)), _mm256_loadu_ps(beta + i + 8));

            return _mm256_add_ps(precompute_values_lower_half, precompute_values_upper_half);
        }

        inline void encode_haswell_8bit(const float *data, uint8_t *codes, int dim, const float *vmin,
                                        const float *vdiff, const float *alpha, const float *beta) {
            int i = 0;
            __m256 precompute_values = _mm256_setzero_ps();
            for (; i + 16 <= dim; i += 16) {
                __m128i ci_vec = encode_haswell(data, i, vmin, vdiff, 256.0f);
                precompute_values = _mm256_add_ps(calc_precomputed_values_haswell(ci_vec, i, alpha, beta),
                                                  precompute_values);
                _mm_storeu_si128((__m128i *) (codes + i), ci_vec);
            }
            double precompute_value = _simsimd_reduce_f32x8_haswell(precompute_values);
            // Handle the remaining
            for (; i < dim; i++) {
                codes[i] = encode_serial(data[i], vmin[i], vdiff[i], 256.0f);
                precompute_value += (double) compute_precomputed_value_serial(codes[i], alpha[i], beta[i]);
            }
            // Store precomputed values
            *reinterpret_cast<float *>(codes + dim) = (float)precompute_value;
        }

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2"))), apply_to = function)
        inline __m512 decode_skylake(const __m128i ci_vec, size_t i, const float *alpha, const float *beta) {
            __m512i ci_vec32 = _mm512_cvtepu8_epi32(ci_vec);
            __m512 ci_vec_float = _mm512_cvtepi32_ps(ci_vec32);

            // Load alpha and beta
            __m512 alpha_vec = _mm512_loadu_ps(alpha + i);
            __m512 beta_vec = _mm512_loadu_ps(beta + i);

            // x = alpha * ci + beta
            __m512 x_vec = _mm512_fmadd_ps(alpha_vec, ci_vec_float, beta_vec);

            return x_vec;
        }

        inline void compute_asym_l2sq_skylake_8bit(const float *x, const uint8_t *y, double *result, size_t dim,
                                              const float *alpha, const float *beta) {
            __m512 d2_vec = _mm512_setzero();
            __m512 y_vec, x_vec, d_vec;
            size_t i = 0;
            for (; i + 16 <= dim; i += 16) {
                x_vec = _mm512_loadu_ps(x + i);
                __m128i codes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y + i));
                y_vec = decode_skylake(codes, i, alpha, beta);
                d_vec = _mm512_sub_ps(x_vec, y_vec);
                d2_vec = _mm512_fmadd_ps(d_vec, d_vec, d2_vec);
            }
            *result = _mm512_reduce_add_ps(d2_vec);
        }

        inline void compute_sym_l2sq_skylake_8bit(const uint8_t *x, const uint8_t *y, double *result, size_t dim,
                                                  const float *alphaSqr) {
            // TODO
        }

        inline void compute_asym_ip_skylake_8bit(const float *x, const uint8_t *y, double *result, size_t dim,
                                                 const float *alpha, const float *beta) {
            __m512 xy_vec = _mm512_setzero();
            __m512 y_vec, x_vec, d_vec;
            size_t i = 0;
            for (; i + 16 <= dim; i += 16) {
                x_vec = _mm512_loadu_ps(x + i);
                __m128i codes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y + i));
                y_vec = decode_skylake(codes, i, alpha, beta);
                xy_vec = _mm512_fmadd_ps(x_vec, y_vec, xy_vec);
            }
            *result = _mm512_reduce_add_ps(xy_vec);
        }

        inline void compute_sym_ip_skylake_8bit(const uint8_t *x, const uint8_t *y, double *result, size_t dim,
                                                const float *alphaSqr) {
            __m512 xy_vec = _mm512_setzero();
            size_t i = 0;
            __m256i x_codes, y_codes;
            __m512i x_codes16, y_codes16, xy;
            __m512 lower_half, upper_half, alphaSqr_vec_low, alphaSqr_vec_high;
            for (; i + 32 <= dim; i += 32) {
                x_codes = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x + i));
                y_codes = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y + i));
                // Convert to 16 bit integers
                x_codes16 = _mm512_cvtepu8_epi16(x_codes);
                y_codes16 = _mm512_cvtepu8_epi16(y_codes);

                // Multiply and add
                xy = _mm512_mullo_epi16(x_codes16, y_codes16);

                // Convert to 32-bit integers
                lower_half = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(xy)));
                upper_half = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(xy, 1)));

                // xy_vec = _mm512_add_ps(xy_vec, lower_half);
                alphaSqr_vec_low = _mm512_loadu_ps(alphaSqr + i);
                alphaSqr_vec_high = _mm512_loadu_ps(alphaSqr + i + 16);
                xy_vec = _mm512_fmadd_ps(lower_half, alphaSqr_vec_low, xy_vec);
                xy_vec = _mm512_fmadd_ps(upper_half, alphaSqr_vec_high, xy_vec);
            }
            *result = _mm512_reduce_add_ps(xy_vec) + reinterpret_cast<const float *>(x + dim)[0] +
                      reinterpret_cast<const float *>(y + dim)[0];
        }
#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SKYLAKE
#endif // SIMSIMD_TARGET_X86
        class SQ8Bit : public Quantizer<uint8_t> {
            static constexpr size_t HISTOGRAM_NUM_BINS = 512;
            static constexpr size_t SCALAR_RANGE = 256;
        public:
            explicit SQ8Bit(int dim, float breakPointDataRatio = 1.0f) : Quantizer(dim, dim + 4),
                                                                       breakPointDataRatio(breakPointDataRatio) {
                vmin = new float[dim];
                vdiff = new float[dim];
                for (size_t i = 0; i < dim; i++) {
                    vmin[i] = std::numeric_limits<float>::max();
                    vdiff[i] = std::numeric_limits<float>::lowest();
                }
                alpha = new float[dim];
                beta = new float[dim];
                alphaSqr = new float[dim];
                betaSqr = new float[dim];

                // initialize the histogram
                histogram = std::vector<std::vector<uint64_t>>(dim);
                for (size_t i = 0; i < dim; i++) {
                    histogram[i] = std::vector<uint64_t>(HISTOGRAM_NUM_BINS);
                }
                for (size_t i = 0; i < dim; i++) {
                    for (size_t j = 0; j < HISTOGRAM_NUM_BINS; j++) {
                        histogram[i][j] = 0;
                    }
                }
            }

            inline void batch_train(size_t n, const float *x) override {
                for (size_t i = 0; i < n; i++) {
                    for (size_t j = 0; j < dim; j++) {
                        vmin[j] = std::min(vmin[j], x[i * dim + j]);
                        vdiff[j] = std::max(vdiff[j], x[i * dim + j]);
                    }
                }
            }

            inline void finalize_train() {
                for (size_t i = 0; i < dim; i++) {
                    vdiff[i] -= vmin[i];
                    alpha[i] = vdiff[i] / SCALAR_RANGE;
                    beta[i] = (0.5f * alpha[i]) + vmin[i];
                    alphaSqr[i] = alpha[i] * alpha[i];
                    betaSqr[i] = beta[i] * beta[i];
                }
            }

            inline void determine_breakpoint(size_t n, const float *data) {
                if (breakPointDataRatio >= 1.0f) {
                    return;
                }
                // Use histogram to determine the smallest break point.
                // We will use 256 bins.
                for (size_t i = 0; i < n; i++) {
                    for (size_t j = 0; j < dim; j++) {
                        // TODO: This should be using simd instruction.
                        // Determine the bin using vmin and vdiff.
                        auto bin = static_cast<uint64_t>(((data[i * dim + j] - vmin[j]) / vdiff[j]) * (float) HISTOGRAM_NUM_BINS);
                        bin = std::min((int) bin, (int) HISTOGRAM_NUM_BINS - 1);
                        histogram[j][bin]++;
                    }
                }
                numTrainedVecs += n;

                // Now we have to find the smallest which contains at-least 70% of n of the data.
                // Find the smallest bin range that contains at least 70% of the data
                auto threshold = n * breakPointDataRatio;
                for (size_t i = 0; i < dim; i++) {
                    size_t start_bin = 0;
                    size_t end_bin = 0;
                    size_t min_bin_size = HISTOGRAM_NUM_BINS;
                    size_t sum = 0;
                    size_t left = 0;

                    // Sliding window approach to find the smallest range
                    for (size_t right = 0; right < HISTOGRAM_NUM_BINS; right++) {
                        sum += histogram[i][right];

                        // Shrink the window from the left if the threshold is met
                        while (sum >= threshold) {
                            if (right - left < min_bin_size) {
                                min_bin_size = right - left;
                                start_bin = left;
                                end_bin = right;
                            }
                            sum -= histogram[i][left];
                            left++;
                        }
                    }
                    vmin[i] = vmin[i] + (float) start_bin / HISTOGRAM_NUM_BINS * vdiff[i];
                    vdiff[i] = (float) (end_bin - start_bin) / HISTOGRAM_NUM_BINS * vdiff[i];
                }
            }

            inline void encode(const float *x, uint8_t *codes, size_t n) const override {
                for (size_t i = 0; i < n; i++) {
                    const float *xi = x + i * dim;
                    // We need to skip the last 4 bytes as they are precomputed values
                    uint8_t *ci = codes + i * codeSize;
#if SIMSIMD_TARGET_HASWELL
                    encode_haswell_8bit(xi, ci, dim, vmin, vdiff, alpha, beta);
#elif SIMSIMD_TARGET_NEON
                    encode_neon_8bit(xi, ci, dim, vmin, vdiff, alpha, beta);
#else
                    encode_serial_8bit(xi, ci, dim, vmin, vdiff, alpha, beta, 0);
#endif
                }
            }

            inline void decode(const uint8_t *codes, float *x, size_t n) const override {
                for (size_t i = 0; i < n; i++) {
                    const uint8_t *ci = codes + i * codeSize;
                    float *xi = x + i * dim;
                    for (size_t j = 0; j < dim; j++) {
                        xi[j] = decode_serial(ci[j], alpha[j], beta[j]);
                    }
                }
            }

            inline float decode_one(const uint8_t code, int d) const {
                return decode_serial(code, alpha[d], beta[d]);
            }

            void flush_to_disk(std::ofstream& out) const {
                // Write the basic fields
                out.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
                out.write(reinterpret_cast<const char *>(&breakPointDataRatio), sizeof(breakPointDataRatio));
                out.write(reinterpret_cast<const char *>(&numTrainedVecs), sizeof(numTrainedVecs));
                // Write the vmin and vdiff
                out.write(reinterpret_cast<const char *>(vmin), dim * sizeof(float));
                out.write(reinterpret_cast<const char *>(vdiff), dim * sizeof(float));
                // Write the alpha and beta
                out.write(reinterpret_cast<const char *>(alpha), dim * sizeof(float));
                out.write(reinterpret_cast<const char *>(beta), dim * sizeof(float));
                // Write the alphaSqr and betaSqr
                out.write(reinterpret_cast<const char *>(alphaSqr), dim * sizeof(float));
                out.write(reinterpret_cast<const char *>(betaSqr), dim * sizeof(float));
                // Write the histogram
                for (size_t i = 0; i < dim; i++) {
                    out.write(reinterpret_cast<const char *>(histogram[i].data()), HISTOGRAM_NUM_BINS * sizeof(uint64_t));
                }
            }

            void load_from_disk(std::ifstream& in) {
                // Read the basic fields
                in.read(reinterpret_cast<char *>(&dim), sizeof(dim));
                in.read(reinterpret_cast<char *>(&breakPointDataRatio), sizeof(breakPointDataRatio));
                in.read(reinterpret_cast<char *>(&numTrainedVecs), sizeof(numTrainedVecs));
                // Read the vmin and vdiff
                in.read(reinterpret_cast<char *>(vmin), dim * sizeof(float));
                in.read(reinterpret_cast<char *>(vdiff), dim * sizeof(float));
                // Read the alpha and beta
                in.read(reinterpret_cast<char *>(alpha), dim * sizeof(float));
                in.read(reinterpret_cast<char *>(beta), dim * sizeof(float));
                // Read the alphaSqr and betaSqr
                in.read(reinterpret_cast<char *>(alphaSqr), dim * sizeof(float));
                in.read(reinterpret_cast<char *>(betaSqr), dim * sizeof(float));
                // Read the histogram
                for (size_t i = 0; i < dim; i++) {
                    histogram[i].resize(HISTOGRAM_NUM_BINS);
                    in.read(reinterpret_cast<char *>(histogram[i].data()), HISTOGRAM_NUM_BINS * sizeof(uint64_t));
                }
            }

            inline const float* getAlpha() const {
                return alpha;
            }

            inline const float* getBeta() const {
                return beta;
            }

            inline const float* getAlphaSqr() const {
                return alphaSqr;
            }

            inline std::unique_ptr<DistanceComputer<float, uint8_t>>
            get_asym_distance_computer(DistanceType type) const override {
                throw std::runtime_error("Not implemented");
            }

            inline std::unique_ptr<DistanceComputer<uint8_t, uint8_t>>
            get_sym_distance_computer(DistanceType type) const override {
                throw std::runtime_error("Not implemented");
            }

            ~SQ8Bit() {
                delete[] vmin;
                delete[] vdiff;
                delete[] alpha;
                delete[] beta;
                delete[] alphaSqr;
                delete[] betaSqr;
            }

        private:
            float *vmin;
            float *vdiff;

            // Precomputed values
            float *alpha;
            float *beta;
            float *alphaSqr;
            float *betaSqr;

            // Training
            float breakPointDataRatio;
            uint64_t numTrainedVecs;
            std::vector<std::vector<uint64_t>> histogram;
        };
    } // namespace sq
} // namespace fastq
