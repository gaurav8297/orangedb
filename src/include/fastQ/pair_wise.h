#pragma once

#include <fastQ/common.h>
#include <fastQ/fastq.h>

namespace fastq {
    namespace pair_wise {
        // TODO: Optimize using SIMD
        inline std::pair<std::vector<uint8_t>, bool>
        encode_8_serial(const float *x, size_t i, const float *vmin, const float *vdiff, const float *skewed_vmin,
                        const float *skewed_vdiff) {
            bool mask = true;
            bool mask2 = false;
            for (size_t j = 0; j < 8; j++) {
                auto k = i + j;
                auto skewed_vmax = skewed_vmin[k] + skewed_vdiff[k];
                mask &= x[k] >= skewed_vmin[k] && x[k] <= skewed_vmax;
                mask2 |= x[k] < skewed_vmin[k] || x[k] > skewed_vmax;
            }

            std::vector<uint8_t> vals(8);
            if (mask) {
                for (size_t j = 0; j < 8; j++) {
                    auto k = i + j;
                    auto scaled = (x[k] - skewed_vmin[k]) / skewed_vdiff[k];
                    vals[j] = std::min(int(scaled * 4.0f), 3);
                    if (vals[j] == 3) {
                        throw std::runtime_error("Invalid value");
                    }
                }
            } else {
                for (size_t j = 0; j < 8; j++) {
                    auto k = i + j;
                    auto diff = std::min(std::max(x[k] - vmin[k], 0.0f), vdiff[k]);
                    auto scaled = diff / vdiff[k];
                    vals[j] = std::min(int(scaled * 4.0f), 3);
                    if (vals[j] == 3) {
                        throw std::runtime_error("Invalid value");
                    }
                }
            }
            return std::make_pair(vals, mask);
        }

        inline std::pair<std::vector<uint8_t>, bool>
        encode_4_serial(const float *x, size_t i, const float *vmin, const float *vdiff, const float *skewed_vmin,
                        const float *skewed_vdiff) {
            bool mask = true;
            for (size_t j = 0; j < 4; j++) {
                auto k = i + j;
                auto skewed_vmax = skewed_vmin[k] + skewed_vdiff[k];
                mask &= (x[k] >= skewed_vmin[k] && x[k] <= skewed_vmax);
            }

            std::vector<uint8_t> vals(4);
            if (mask) {
                for (size_t j = 0; j < 4; j++) {
                    auto k = i + j;
                    auto scaled = (x[k] - skewed_vmin[k]) / skewed_vdiff[k];
                    vals[j] = std::min(int(scaled * 4.0f), 3);
                }
            } else {
                for (size_t j = 0; j < 4; j++) {
                    auto k = i + j;
                    auto diff = std::min(std::max(x[k] - vmin[k], 0.0f), vdiff[k]);
                    auto scaled = diff / vdiff[k];
                    vals[j] = std::min(int(scaled * 4.0f), 3);
                }
            }
            return std::make_pair(vals, mask);
        }

        inline void pack_8_bools_to_uint8(uint8_t *codes, int i, const uint8_t *mask) {
            codes[i] = mask[0] << 0 | mask[1] << 1 | mask[2] << 2 | mask[3] << 3 | mask[4] << 4 | mask[5] << 5 |
                       mask[6] << 6 | mask[7] << 7;
        }

        inline void unpack_uint8_to_8_bool(const uint8_t *codes, int i, uint8_t *mask) {
            mask[0] = codes[i] >> 0 & 1;
            mask[1] = codes[i] >> 1 & 1;
            mask[2] = codes[i] >> 2 & 1;
            mask[3] = codes[i] >> 3 & 1;
            mask[4] = codes[i] >> 4 & 1;
            mask[5] = codes[i] >> 5 & 1;
            mask[6] = codes[i] >> 6 & 1;
            mask[7] = codes[i] >> 7 & 1;
        }

        inline void
        encode_serial(const float *x, uint8_t *codes, size_t n, int dim, int encoded_bytes, const float *vmin,
                      const float *vdiff, const float *skewed_vmin,
                      const float *skewed_vdiff) {
            auto mask_size = ceil(dim / 4.0f);
            std::vector<uint8_t> masks(mask_size);
            for (size_t i = 0; i < n; i++) {
                auto *xi = x + i * dim;
                auto *ci = codes + i * encoded_bytes;
                auto m = 0;
                for (size_t j = 0; j < dim; j += 4) {
                    auto encoded_data = encode_4_serial(xi, j, vmin, vdiff, skewed_vmin, skewed_vdiff);
                    auto data = encoded_data.first;
                    for (size_t k = 0; k < 4; k++) {
                        ci[j + k] = data[k];
                    }
                    masks[m] = (uint8_t) encoded_data.second;
                    m++;
                }
                int k = 0;
                for (size_t j = 0; j < mask_size; j += 8) {
                    pack_8_bools_to_uint8(ci + dim, k, masks.data() + j);
                    k++;
                }
            }
        }

#if SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("+simd")
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)

        inline float32x4x2_t
        decode_neon(const uint8_t *code, size_t i, const float *alpha, const float *beta, const float *skewed_alpha,
                    const float *skewed_beta, uint8_t mask1, uint8_t mask2) {
            uint8x8_t ci_vec = vld1_u8(code + i);
            uint16x8_t ci_vec16 = vmovl_u8(ci_vec);
            float32x4_t ci_vec32_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(ci_vec16)));
            float32x4_t ci_vec32_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(ci_vec16)));

            // x = alpha * ci + beta
            float32x4_t x_low = vmlaq_f32(vld1q_f32(beta + i), vld1q_f32(alpha + i), ci_vec32_low);
            float32x4_t x_high = vmlaq_f32(vld1q_f32(beta + i + 4), vld1q_f32(alpha + i + 4), ci_vec32_high);

            // skewed_x = skewed_alpha * ci + skewed_beta
            float32x4_t skewed_x_low = vmlaq_f32(vld1q_f32(skewed_beta + i), vld1q_f32(skewed_alpha + i), ci_vec32_low);
            float32x4_t skewed_x_high = vmlaq_f32(vld1q_f32(skewed_beta + i + 4), vld1q_f32(skewed_alpha + i + 4),
                                                  ci_vec32_high);

            float32x4_t low = vbslq_f32(vceqq_s32(vdupq_n_s32(mask1), vdupq_n_s32(1)), skewed_x_low, x_low);
            float32x4_t high = vbslq_f32(vceqq_s32(vdupq_n_s32(mask2), vdupq_n_s32(1)), skewed_x_high, x_high);
            return {low, high};
        }

        inline void decode_neon(const uint8_t *code, float *x, size_t n, size_t dim, int codeSize, const float *alpha,
                                const float *beta,
                                const float *skewed_alpha, const float *skewed_beta) {
            auto code_mask_size = ceil(dim / 32.0f);
            std::vector<uint8_t> masks(dim / 4);
            for (int i = 0; i < n; i++) {
                auto ci = code + i * codeSize;
                auto xi = x + i * dim;
                for (int j = 0; j < code_mask_size; j++) {
                    unpack_uint8_to_8_bool(ci + dim, j, masks.data() + j * 8);
                }
                int k = 0;
                simsimd_size_t m = 0;
                for (; m + 8 <= dim; m += 8) {
                    float32x4x2_t x_decoded = decode_neon(ci, m, alpha, beta, skewed_alpha, skewed_beta, masks[k],
                                                          masks[k + 1]);
                    vst1q_f32(xi + m, x_decoded.val[0]);
                    vst1q_f32(xi + m + 4, x_decoded.val[1]);
                    k += 2;
                }
            }
        }

        inline void compute_asym_l2sqr_neon(const float *x, const uint8_t *y, double *result, size_t dim,
                                            const float *alpha, const float *beta, const float *skewed_alpha,
                                            const float *skewed_beta, int code_mask_size, uint8_t *masks) {
            float32x4_t sum_vec = vdupq_n_f32(0);
            for (int i = 0; i < code_mask_size; i++) {
                unpack_uint8_to_8_bool(y + dim, i, masks + i * 8);
            }
            int k = 0;
            simsimd_size_t i = 0;
            for (; i + 8 <= dim; i += 8) {
                float32x4x2_t y_decoded = decode_neon(y, i, alpha, beta, skewed_alpha, skewed_beta, masks[k],
                                                      masks[k + 1]);
                float32x4_t x_low_vec = vld1q_f32(x + i);
                float32x4_t y_low_vec = y_decoded.val[0];
                float32x4_t x_high_vec = vld1q_f32(x + i + 4);
                float32x4_t y_high_vec = y_decoded.val[1];

                float32x4_t diff_low = vsubq_f32(x_low_vec, y_low_vec);
                sum_vec = vfmaq_f32(sum_vec, diff_low, diff_low);
                float32x4_t diff_high = vsubq_f32(x_high_vec, y_high_vec);
                sum_vec = vfmaq_f32(sum_vec, diff_high, diff_high);
                k += 2;
            }
            *result = vaddvq_f32(sum_vec);
        }

        inline void compute_sym_l2sqr_neon(const uint8_t *x, const uint8_t *y, double *result, size_t dim,
                                           const float *alpha, const float *beta, const float *skewed_alpha,
                                           const float *skewed_beta, int code_mask_size, uint8_t *x_masks,
                                           uint8_t *y_masks) {
            float32x4_t sum_vec = vdupq_n_f32(0);
            for (int i = 0; i < code_mask_size; i++) {
                unpack_uint8_to_8_bool(x + dim, i, x_masks + i * 8);
                unpack_uint8_to_8_bool(y + dim, i, y_masks + i * 8);
            }
            int k = 0;
            simsimd_size_t i = 0;
            for (; i + 8 <= dim; i += 8) {
                float32x4x2_t x_decoded = decode_neon(x, i, alpha, beta, skewed_alpha, skewed_beta, x_masks[k],
                                                      x_masks[k + 1]);
                float32x4x2_t y_decoded = decode_neon(y, i, alpha, beta, skewed_alpha, skewed_beta, y_masks[k],
                                                      y_masks[k + 1]);
                float32x4_t x_low_vec = x_decoded.val[0];
                float32x4_t y_low_vec = y_decoded.val[0];
                float32x4_t x_high_vec = x_decoded.val[1];
                float32x4_t y_high_vec = y_decoded.val[1];

                float32x4_t diff_low = vsubq_f32(x_low_vec, y_low_vec);
                sum_vec = vfmaq_f32(sum_vec, diff_low, diff_low);

                float32x4_t diff_high = vsubq_f32(x_high_vec, y_high_vec);
                sum_vec = vfmaq_f32(sum_vec, diff_high, diff_high);
                k += 2;
            }
            *result = vaddvq_f32(sum_vec);
        }

#pragma clang attribute pop
#pragma GCC pop_options
#endif
#endif

        class AsymmetricL2Sq : public DistanceComputer<float, uint8_t> {
        public:
            explicit AsymmetricL2Sq(int dim, const float *alpha, const float *beta, const float *skewed_alpha,
                                    const float *skewed_beta)
                    : DistanceComputer(dim), alpha(alpha), beta(beta), skewed_alpha(skewed_alpha),
                      skewed_beta(skewed_beta) {
                masks = std::vector<uint8_t>(dim / 4);
                code_mask_size = ceil(dim / 32.0f);
            };

            inline void compute_distance(const float *x, const uint8_t *y, double *result) override {
#if SIMSIMD_TARGET_NEON
                compute_asym_l2sqr_neon(x, y, result, dim, alpha, beta, skewed_alpha, skewed_beta, code_mask_size,
                                        masks.data());
#else
                throw std::runtime_error("Not implemented");
#endif
            }

            inline void batch_compute_distances(const float *x, const uint8_t *y, double *results, size_t n) override {
                throw std::runtime_error("Not implemented");
            }

        private:
            const float *alpha;
            const float *beta;
            const float *skewed_alpha;
            const float *skewed_beta;

            std::vector<uint8_t> masks;
            int code_mask_size;
        };

        class SymmetricL2Sq : public DistanceComputer<uint8_t, uint8_t> {
        public:
            explicit SymmetricL2Sq(int dim, const float *alpha, const float *beta, const float *skewed_alpha,
                                   const float *skewed_beta)
                    : DistanceComputer(dim), alpha(alpha), beta(beta), skewed_alpha(skewed_alpha),
                      skewed_beta(skewed_beta) {
                x_masks = std::vector<uint8_t>(dim / 4);
                y_masks = std::vector<uint8_t>(dim / 4);
                code_mask_size = ceil(dim / 32.0f);
            };

            ~SymmetricL2Sq() = default;

            inline void compute_distance(const uint8_t *x, const uint8_t *y, double *result) override {
#if SIMSIMD_TARGET_NEON
                compute_sym_l2sqr_neon(x, y, result, dim, alpha, beta, skewed_alpha, skewed_beta, code_mask_size,
                                       x_masks.data(), y_masks.data());
#else
                throw std::runtime_error("Not implemented");
#endif
            }

            inline void
            batch_compute_distances(const uint8_t *x, const uint8_t *y, double *results, size_t n) override {
                throw std::runtime_error("Not implemented");
            }

        private:
            const float *alpha;
            const float *beta;
            const float *skewed_alpha;
            const float *skewed_beta;

            std::vector<uint8_t> x_masks;
            std::vector<uint8_t> y_masks;
            int code_mask_size;
        };

        // TODO: Maybe we can use histogram to find the break points. That might be more accurate. The advantage
        //  is we can utilize data into consideration.
        class PairWise2Bit : public Quantizer<uint8_t> {
            static constexpr size_t NUM_BINS = 512;
            static constexpr float BREAK_POINT_DATA_RATIO = 0.8f;
            static constexpr float SCALAR_DATA_RATIO = 0.95f;
        public:
            explicit PairWise2Bit(int dim)
                    : Quantizer<uint8_t>(dim, dim + ceil(dim / 32.0f)) {
                vmin = new float[dim];
                vdiff = new float[dim];
                skewed_vmin = new float[dim];
                skewed_vdiff = new float[dim];
                skewed_alpha = new float[dim];
                skewed_beta = new float[dim];
                non_skewed_alpha = new float[dim];
                non_skewed_beta = new float[dim];

                for (size_t i = 0; i < dim; i++) {
                    vmin[i] = std::numeric_limits<float>::max();
                    vdiff[i] = std::numeric_limits<float>::lowest();
                }
            };

            inline void get_breaking_points(const std::vector<std::vector<uint64_t>> &histogram,
                                            float threshold, size_t d, size_t &start_bin, size_t &end_bin) {
                size_t min_bin_size = NUM_BINS;
                size_t sum = 0;
                size_t left = 0;
                start_bin = 0;
                end_bin = 0;
                // Sliding window approach to find the smallest range
                for (size_t right = 0; right < NUM_BINS; right++) {
                    sum += histogram[d][right];

                    // Shrink the window from the left if the threshold is met
                    while (sum >= threshold) {
                        if (right - left < min_bin_size) {
                            min_bin_size = right - left;
                            start_bin = left;
                            end_bin = right;
                        }
                        sum -= histogram[d][left];
                        left++;
                    }
                }
            }

            inline void determine_smallest_breakpoint(size_t n, const float *data) {
                printf("Calculating Histogram!!\n");
                // Use histogram to determine the smallest break point.
                // We will use 256 bins.
                std::vector<std::vector<uint64_t>> histogram(dim);
                for (size_t i = 0; i < dim; i++) {
                    histogram[i].resize(NUM_BINS, 0);
                }

                for (size_t i = 0; i < n; i++) {
                    for (size_t j = 0; j < dim; j++) {
                        // Determine the bin using vmin and vdiff.
                        auto bin = static_cast<uint64_t>(((data[i * dim + j] - vmin[j]) / vdiff[j]) * NUM_BINS);
                        bin = std::min((int) bin, (int) NUM_BINS - 1);
                        histogram[j][bin]++;
                    }
                }

                // Now we have to find the smallest which contains at-least 70% of n of the data.
                // Find the smallest bin range that contains at least 70% of the data
                auto skewed_threshold = n * BREAK_POINT_DATA_RATIO;
                auto non_skewed_threshold = n * SCALAR_DATA_RATIO;
                for (size_t i = 0; i < dim; i++) {
                    size_t skewed_start_bin, skewed_end_bin, start_bin, end_bin;
                    get_breaking_points(histogram, skewed_threshold, i, skewed_start_bin, skewed_end_bin);
                    get_breaking_points(histogram, non_skewed_threshold, i, start_bin, end_bin);

                    skewed_vmin[i] = vmin[i] + ((float) skewed_start_bin / NUM_BINS) * vdiff[i];
                    skewed_vdiff[i] = ((float) (skewed_end_bin - skewed_start_bin) / NUM_BINS) * vdiff[i];

                    vmin[i] = vmin[i] + ((float) start_bin / NUM_BINS) * vdiff[i];
                    vdiff[i] = (float) (end_bin - start_bin) / NUM_BINS * vdiff[i];
                }
            }

            inline void compute_alpha_beta() {
                for (size_t i = 0; i < dim; i++) {
                    skewed_alpha[i] = skewed_vdiff[i] / 4.0f;
                    skewed_beta[i] = 0.5f * skewed_alpha[i] + skewed_vmin[i];
                    non_skewed_alpha[i] = vdiff[i] / 4.0f;
                    non_skewed_beta[i] = 0.5f * non_skewed_alpha[i] + vmin[i];
                }
            }

            inline void batch_train(size_t n, const float *x) override {
                for (size_t i = 0; i < n; i++) {
                    for (size_t j = 0; j < dim; j++) {
                        vmin[j] = std::min(vmin[j], x[i * dim + j]);
                        vdiff[j] = std::max(vdiff[j], x[i * dim + j]);
                    }
                }
                for (size_t i = 0; i < dim; i++) {
                    vdiff[i] -= vmin[i];
                }

                // Print vmin and vdiff
                for (size_t i = 0; i < 5; i++) {
                    printf("vmin[%ld]: %f, vdiff[%ld]: %f\n", i, vmin[i], i, vdiff[i]);
                }
                determine_smallest_breakpoint(n, x);
                for (size_t i = 0; i < 5; i++) {
                    printf("vmin[%ld]: %f, vdiff[%ld]: %f\n", i, vmin[i], i, vdiff[i]);
                }
                for (size_t i = 0; i < 5; i++) {
                    printf("skewed_vmin[%ld]: %f, skewed_vdiff[%ld]: %f\n", i, skewed_vmin[i], i, skewed_vdiff[i]);
                }
                compute_alpha_beta();
            }

            inline void encode(const float *x, uint8_t *codes, size_t n) const override {
                encode_serial(x, codes, n, dim, codeSize, vmin, vdiff, skewed_vmin, skewed_vdiff);
            }

            inline void decode(const uint8_t *code, float *x, size_t n) const override {
                decode_neon(code, x, n, dim, codeSize, non_skewed_alpha, non_skewed_beta, skewed_alpha, skewed_beta);
            }

            inline std::unique_ptr<DistanceComputer<float, uint8_t>>
            get_asym_distance_computer(DistanceType type) const override {
                return std::make_unique<AsymmetricL2Sq>(dim, non_skewed_alpha, non_skewed_beta, skewed_alpha,
                                                        skewed_beta);
            }

            inline std::unique_ptr<DistanceComputer<uint8_t, uint8_t>>
            get_sym_distance_computer(DistanceType type) const override {
                return std::make_unique<SymmetricL2Sq>(dim, non_skewed_alpha, non_skewed_beta, skewed_alpha,
                                                       skewed_beta);
            }

            ~PairWise2Bit() {
                delete[] vmin;
                delete[] vdiff;
                delete[] non_skewed_alpha;
                delete[] non_skewed_beta;
                delete[] skewed_vmin;
                delete[] skewed_vdiff;
                delete[] skewed_alpha;
                delete[] skewed_beta;
            }

        public:
            // Non skewed part information
            float *vmin;
            float *vdiff;
            float *non_skewed_alpha;
            float *non_skewed_beta;

            // Skewed part information
            float *skewed_vmin;
            float *skewed_vdiff;
            float *skewed_alpha;
            float *skewed_beta;
        };
    } // namespace pair_wise
} // namespace fastQ
