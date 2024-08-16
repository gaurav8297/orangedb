#pragma once

#include <stdint.h>
#include <cstdio>

namespace orangedb {
    namespace quantizer {

        // *  For hardware architectures:
        // *  - Arm (NEON, SVE)
        // *  - x86 (AVX2, AVX512)

        enum DistanceType {
            L2,
            Cosine,
            InnerProduct
        };

        struct Quantizer {
            explicit Quantizer(size_t dim = 0, size_t codeSize = 0)
                    : dim(dim), codeSize(codeSize) {};

            /**
             * Train the quantizer
             * @param n number of training vectors
             * @param x training vectors, size n * d
             */
            virtual void batchTrain(size_t n, const float *x) = 0;

            /**
             * Quantize a set of vectors
             * @param x input vectors, size n * d
             * @param codes output codes, size n * code_size
             */
            virtual void encode(const float *x, uint8_t *codes, size_t n) const = 0;

            /**
             * Decode a set of vectors
             * @param code input codes, size n * code_size
             * @param x output vectors, size n * d
             */
            virtual void decode(const uint8_t *code, float *x, size_t n) const = 0;

            /**
             * Get the distance computer for the given metric
             * @param type the distance metric
             * @return the distance computer
             */
            virtual std::unique_ptr<DistanceComputer> getDistanceComputer(DistanceType type) const = 0;

            size_t dim; // dimension of the input vectors
            size_t codeSize; // bytes per indexed vector
        };

        // TODO: Implement Scalar 8bit quantization
        // TODO: Implement Scalar 4bit quantization

        class ScalarQuantizer8Bit : public Quantizer {
        public:
            explicit ScalarQuantizer8Bit(size_t dim) : Quantizer(dim, 8) {
                vmin = new float[dim];
                vdiff = new float[dim];
                vdiffSqr = new float[dim];
                for (size_t i = 0; i < dim; i++) {
                    vmin[i] = std::numeric_limits<float>::max();
                    vdiff[i] = std::numeric_limits<float>::lowest();
                }

                for (size_t i = 0; i < dim; i++) {
                    precomputedDistTable.push_back(new float[256]);
                }
            }

            inline void batchTrain(size_t n, const float *x) {
                for (size_t i = 0; i < n; i++) {
                    for (size_t j = 0; j < dim; j++) {
                        vmin[j] = std::min(vmin[j], x[i * dim + j]);
                        vdiff[j] = std::max(vdiff[j], x[i * dim + j]);
                    }
                }

                for (size_t i = 0; i < dim; i++) {
                    vdiff[i] -= vmin[i];
                }

                // Precompute the distance table
                precomputeDistanceTable();
            }

            inline void encode(const float *x, uint8_t *codes, size_t n) const {
                for (size_t i = 0; i < n; i++) {
                    const float *xi = x + i * dim;
                    uint8_t *ci = codes + i * dim;
                    for (size_t j = 0; j < dim; j++) {
                        // Scale to [0, 1]
                        float descaled_val = (xi[j] - vmin[j]) / vdiff[j];
                        // Scale to [0, 255]
                        ci[j] = int(descaled_val * 255.0f);
                    }
                }
            }

            inline void decode(const uint8_t *code, float *x, size_t n) const {
                for (size_t i = 0; i < n; i++) {
                    const uint8_t *ci = code + i * dim;
                    float *xi = x + i * dim;
                    for (size_t j = 0; j < dim; j++) {
                        // Scale to [0, 1]. Reason to add 0.5f is to round the value.
                        // This is used for continuity correction. Basically the probability that a random variable
                        // falls in a certain range is the same as the probability that the random variable falls in the
                        // corresponding integer value. It gives out a more accurate result.
                        float scaled_val = (ci[j] + 0.5f) / 255.0f;
                        xi[j] = vmin[j] + scaled_val * vdiff[j];
                    }
                }
            }

            std::unique_ptr<DistanceComputer> getDistanceComputer(DistanceType type) const override;

            ~ScalarQuantizer8Bit() {
                delete[] vmin;
                delete[] vdiff;
                for (size_t i = 0; i < dim; i++) {
                    delete[] precomputedDistTable[i];
                }
            }

        private:
            inline void precomputeDistanceTable() {
                for (size_t i = 0; i < dim; i++) {
                    for (int j = 0; j < 256; j++) {
                        precomputedDistTable[i][j] = j * vdiff[i] * vmin[i];
                    }
                    vdiffSqr[i] = vdiff[i] * vdiff[i];
                }
            }

        private:
            float *vmin;
            float *vdiff;

            // Precomputed distance variables
            std::vector<float *> precomputedDistTable;
            float *vdiffSqr;
        };
    }
} // namespace orangedb
