#pragma once

#include <vector>
#include <simsimd/simsimd.h>

namespace fastq {
    // TODO: Probably get rid of virtual functions and use template specialization. Benchmark it!!

    enum DistanceType {
        L2,
        INNER_PRODUCT,
        COSINE,
    };

    template<typename T1, typename T2>
    struct DistanceComputer {
        int dim;

        explicit DistanceComputer(int dim) : dim(dim) {}

        /**
         * Compute the distance between two vectors.
         * @param x the first vector
         * @param y the second vector
         * @param result the distance
         */
        virtual void compute_distance(const T1 *x, const T2 *y, double *result) = 0;

        /**
         * Compute the asymmetric batch distances between n vectors. More accurate but slower.
         * @param x the first set of vectors
         * @param y the second set of vectors
         * @param result the distances
         * @param n the number of vectors
         */
        virtual void batch_compute_distances(const T1 *x, const T2 *y, double *results, size_t n) = 0;

        virtual ~DistanceComputer() = default;
    };

    struct FlatL2DistanceComputer : public DistanceComputer<float, float> {
        explicit FlatL2DistanceComputer(int dim) : DistanceComputer(dim) {}

        void compute_distance(const float *x, const float *y, double *result) override {
            simsimd_l2sq_f32(x, y, dim, result);
        }

        void batch_compute_distances(const float *x, const float *y, double *results, size_t n) override {
            for (size_t i = 0; i < n; i++) {
                simsimd_l2sq_f32(x + i * dim, y + i * dim, dim, results + i);
            }
        }
    };

    template<typename T>
    struct Quantizer {
        explicit Quantizer(int dim = 0, size_t codeSize = 0)
                : dim(dim), codeSize(codeSize) {};

        /**
         * Train the quantizer
         * @param n number of training vectors
         * @param x training vectors, size n * d
         */
        virtual void batch_train(size_t n, const float *x) = 0;

        /**
         * Quantize a set of vectors
         * @param x input vectors, size n * d
         * @param codes output codes, size n * code_size
         */
        virtual void encode(const float *x, T *codes, size_t n) const = 0;

        /**
         * Decode a set of vectors
         * @param code input codes, size n * code_size
         * @param x output vectors, size n * d
         */
        virtual void decode(const T *code, float *x, size_t n) const = 0;

        /**
         * Get the asymmetric distance computer which computes the distance between a quantized
         * vector and a actual float vector. This is more accurate but slower.
         */
        virtual std::unique_ptr<DistanceComputer<float, T>> get_asym_distance_computer(DistanceType type) const = 0;

        /**
         * Get the symmetric distance computer which computes the distance between two quantized vectors.
         * This is faster.
         */
        virtual std::unique_ptr<DistanceComputer<T, T>> get_sym_distance_computer(DistanceType type) const = 0;

        int dim; // dimension of the input vectors
        size_t codeSize; // bytes per indexed vector
    };
} // namespace fastq

