#pragma once

#include <cassert>
#include <unistd.h>
#include <unordered_set>
#include <memory>
#include <simsimd/simsimd.h>
#include <common.h>
#include <fastQ/scalar_8bit.h>

using namespace std;
using namespace fastq::scalar_8bit;

namespace orangedb {
    enum DistanceType {
        L2,
        IP,
        COSINE,
    };

    template<typename T>
    struct DistanceComputer {
        int dim;

        explicit DistanceComputer(int dim) : dim(dim) {
        }

        virtual void computeDistance(const T *dest, double *result) = 0;

        virtual void computeDistance(const T *src, const T *dest, double *result) = 0;

        virtual void setQuery(const float *query) = 0;

        virtual std::unique_ptr<DistanceComputer> clone() = 0;

        virtual ~DistanceComputer() = default;
    };

    // struct DistanceComputer {
    //     virtual void computeDistance(vector_idx_t id, double *result) = 0;
    //
    //     virtual void computeDistance(vector_idx_t src, vector_idx_t dest, double *result) = 0;
    //
    //     virtual void computeDistance(vector_idx_t src, vector_idx_t dest, int dim, double *result) = 0;
    //
    //     virtual void batchComputeDistances(vector_idx_t *ids, double *results, int size) = 0;
    //
    //     virtual void computeDistance(float* src, float* dest, double *result) = 0;
    //
    //     virtual void setQuery(const float *query) = 0;
    //
    //     virtual std::unique_ptr<DistanceComputer> clone() = 0;
    //
    //     virtual ~DistanceComputer() = default;
    // };

    struct L2DistanceComputer : public DistanceComputer<float> {
        explicit L2DistanceComputer(int dim)
            : DistanceComputer<float>(dim) {
        }

        inline void computeDistance(const float *dest, double *result) override {
            simsimd_l2sq_f32(query, dest, dim, result);
        }

        inline void computeDistance(const float *src, const float *dest, double *result) override {
            simsimd_l2sq_f32(src, dest, dim, result);
        }

        inline void setQuery(const float *query) override {
            this->query = query;
        }

        inline std::unique_ptr<DistanceComputer<float>> clone() override {
            return std::make_unique<L2DistanceComputer>(dim);
        }

        ~L2DistanceComputer() override = default;

    private:
        const float *query;
    };

    struct IPDistanceComputer : public DistanceComputer<float> {
        explicit IPDistanceComputer(int dim) : DistanceComputer<float>(dim) {
        }

        inline void computeDistance(const float *dest, double *result) override {
            simsimd_dot_f32(query, dest, dim, result);
        }

        inline void computeDistance(const float *src, const float *dest, double *result) override {
            simsimd_dot_f32(src, dest, dim, result);
        }

        inline void setQuery(const float *query) override {
            this->query = query;
        }

        inline std::unique_ptr<DistanceComputer<float>> clone() override {
            return std::make_unique<IPDistanceComputer>(dim);
        }

        ~IPDistanceComputer() override = default;

    private:
        const float *query;
    };

    struct CosineDistanceComputer : public DistanceComputer<float> {
        explicit CosineDistanceComputer(int dim) : DistanceComputer<float>(dim) {
        }

        inline void computeDistance(const float *dest, double *result) override {
            simsimd_cos_f32(query, dest, dim, result);
        }

        inline void computeDistance(const float *src, const float *dest, double *result) override {
            simsimd_cos_f32(src, dest, dim, result);
        }

        inline void setQuery(const float *query) override {
            this->query = query;
        }

        inline std::unique_ptr<DistanceComputer<float>> clone() override {
            return std::make_unique<CosineDistanceComputer>(dim);
        }

        ~CosineDistanceComputer() override = default;

    private:
        const float *query;
    };

    class SQ8AsymL2Sq : public DistanceComputer<uint8_t> {
    public:
        explicit SQ8AsymL2Sq(int dim, const float *alpha, const float *beta, const float *alphaSqr)
            : DistanceComputer(dim), alpha(alpha), beta(beta), alphaSqr(alphaSqr) {
        };

        ~SQ8AsymL2Sq() override = default;

        inline void computeDistance(const uint8_t *dest, double *result) override {
#if SIMSIMD_TARGET_SKYLAKE
        compute_asym_l2sq_skylake_8bit(query, dest, result, dim, alpha, beta);
#else
            compute_asym_l2sq_serial_8bit(query, dest, result, dim, alpha, beta);
#endif
        }

        inline void computeDistance(const uint8_t *src, const uint8_t *dest, double *result) override {
#if SIMSIMD_TARGET_SKYLAKE
        compute_sym_l2sq_skylake_8bit(src, dest, result, dim, alphaSqr);
#else
            compute_sym_l2sq_serial_8bit(src, dest, result, dim, alphaSqr);
#endif
        }

        inline void setQuery(const float *query) override {
            this->query = query;
        }

        inline std::unique_ptr<DistanceComputer<uint8_t>> clone() override {
            return std::make_unique<SQ8AsymL2Sq>(dim, alpha, beta, alphaSqr);
        }

    private:
        const float *alpha;
        const float *beta;
        const float *alphaSqr;

        const float *query;
    };

    class SQ8SymL2Sq : public DistanceComputer<uint8_t> {
    public:
        explicit SQ8SymL2Sq(int dim, const SQ8Bit *quantizer)
            : DistanceComputer(dim), quantizer(quantizer) {
            quantizedQuery = new uint8_t[quantizer->codeSize];
        };

        ~SQ8SymL2Sq() override {
            delete[] quantizedQuery;
        }

        inline void computeDistance(const uint8_t *dest, double *result) override {
#if SIMSIMD_TARGET_SKYLAKE
        compute_sym_l2sq_skylake_8bit(quantizedQuery, dest, result, dim, quantizer->getAlphaSqr());
#else
            compute_sym_l2sq_serial_8bit(quantizedQuery, dest, result, dim, quantizer->getAlphaSqr());
#endif
        }

        inline void computeDistance(const uint8_t *src, const uint8_t *dest, double *result) override {
#if SIMSIMD_TARGET_SKYLAKE
        compute_sym_l2sq_skylake_8bit(src, dest, result, dim, quantizer->getAlphaSqr());
#else
            compute_sym_l2sq_serial_8bit(src, dest, result, dim, quantizer->getAlphaSqr());
#endif
        }

        inline void setQuery(const float *query) override {
            quantizer->encode(query, quantizedQuery, 1);
        }

        inline std::unique_ptr<DistanceComputer<uint8_t>> clone() override {
            return std::make_unique<SQ8SymL2Sq>(dim, quantizer);
        }

    private:
        const SQ8Bit *quantizer;
        uint8_t *quantizedQuery;
    };

    class SQ8AsymIP : public DistanceComputer<uint8_t> {
    public:
        explicit SQ8AsymIP(int dim, const float *alpha, const float *beta, const float *alphaSqr)
            : DistanceComputer(dim), alpha(alpha), beta(beta), alphaSqr(alphaSqr) {
        };

        inline void computeDistance(const uint8_t *dest, double *result) override {
            compute_asym_ip_serial_8bit(query, dest, result, dim, alpha, beta);
        }

        inline void computeDistance(const uint8_t *src, const uint8_t *dest, double *result) override {
            compute_sym_ip_serial_8bit(src, dest, result, dim, alphaSqr);
        }

        inline void setQuery(const float *query) override {
            this->query = query;
        }

        inline std::unique_ptr<DistanceComputer<uint8_t>> clone() override {
            return std::make_unique<SQ8AsymIP>(dim, alpha, beta, alphaSqr);
        }

    private:
        const float *alpha;
        const float *beta;
        const float *alphaSqr;

        const float *query;
    };

    class SQ8SymIP : public DistanceComputer<uint8_t> {
    public:
        explicit SQ8SymIP(int dim, const SQ8Bit *quantizer)
            : DistanceComputer(dim), quantizer(quantizer) {
            quantizedQuery = new uint8_t[quantizer->codeSize];
        };

        ~SQ8SymIP() override {
            delete[] quantizedQuery;
        }

        inline void computeDistance(const uint8_t *dest, double *result) override {
            compute_sym_ip_serial_8bit(quantizedQuery, dest, result, dim, quantizer->getAlphaSqr());
        }

        inline void computeDistance(const uint8_t *src, const uint8_t *dest, double *result) override {
            compute_sym_ip_serial_8bit(src, dest, result, dim, quantizer->getAlphaSqr());
        }

        inline void setQuery(const float *query) override {
            quantizer->encode(query, quantizedQuery, 1);
        }

        inline std::unique_ptr<DistanceComputer<uint8_t>> clone() override {
            return std::make_unique<SQ8SymIP>(dim, quantizer);
        }

    private:
        const SQ8Bit *quantizer;
        uint8_t *quantizedQuery;
    };

    class SQ8AsymCosine : public DistanceComputer<uint8_t> {
    public:
        explicit SQ8AsymCosine(int dim, const float *alpha, const float *beta, const float *alphaSqr)
            : DistanceComputer(dim), alpha(alpha), beta(beta), alphaSqr(alphaSqr) {
            query = new float[dim];
        };

        ~SQ8AsymCosine() override {
            delete[] query;
        }

        inline void computeDistance(const uint8_t *dest, double *result) override {
#if SIMSIMD_TARGET_SKYLAKE
        compute_asym_ip_skylake_8bit(query, dest, result, dim, alpha, beta);
#else
            compute_asym_ip_serial_8bit(query, dest, result, dim, alpha, beta);
#endif
            *result = 1 - *result;
            *result = *result < 0 ? 0 : *result;
        }

        inline void computeDistance(const uint8_t *src, const uint8_t *dest, double *result) override {
            compute_sym_ip_serial_8bit(src, dest, result, dim, alphaSqr);
            *result = 1 - *result;
            *result = *result < 0 ? 0 : *result;
        }

        inline void setQuery(const float *query) override {
            normalize_vector(query, dim, this->query);
        }

        inline std::unique_ptr<DistanceComputer<uint8_t>> clone() override {
            return std::make_unique<SQ8AsymCosine>(dim, alpha, beta, alphaSqr);
        }

    private:
        const float *alpha;
        const float *beta;
        const float *alphaSqr;
        float *query;
    };

    class SQ8SymCosine : public DistanceComputer<uint8_t> {
    public:
        explicit SQ8SymCosine(int dim, const SQ8Bit *quantizer)
            : DistanceComputer(dim), quantizer(quantizer) {
            normalizedQuery = new float[dim];
            quantizedQuery = new uint8_t[quantizer->codeSize];
        };

        ~SQ8SymCosine() override {
            delete[] normalizedQuery;
            delete[] quantizedQuery;
        }

        inline void computeDistance(const uint8_t *dest, double *result) override {
#if SIMSIMD_TARGET_SKYLAKE
        compute_sym_ip_skylake_8bit(quantizedQuery, dest, result, dim, quantizer->getAlphaSqr());
#else
            compute_sym_ip_serial_8bit(quantizedQuery, dest, result, dim, quantizer->getAlphaSqr());
#endif
            *result = 1 - *result;
            *result = *result < 0 ? 0 : *result;
        }

        inline void computeDistance(const uint8_t *src, const uint8_t *dest, double *result) override {
            compute_sym_ip_serial_8bit(src, dest, result, dim, quantizer->getAlphaSqr());
            *result = 1 - *result;
            *result = *result < 0 ? 0 : *result;
        }

        inline void setQuery(const float *query) override {
            normalize_vector(query, dim, normalizedQuery);
            quantizer->encode(normalizedQuery, quantizedQuery, 1);
        }

        inline std::unique_ptr<DistanceComputer<uint8_t>> clone() override {
            return std::make_unique<SQ8SymCosine>(dim, quantizer);
        }

    private:
        const SQ8Bit *quantizer;
        float *normalizedQuery;
        uint8_t *quantizedQuery;
    };

    template<typename T>
    class DelegateDC {
    public:
        explicit DelegateDC(std::unique_ptr<DistanceComputer<T>> dc, const T *data, int dim, int n)
            : dc(std::move(dc)), data(data), dim(dim), n(n) {
        }

        inline void computeDistance(vector_idx_t id, double *result) {
            CHECK_ARGUMENT(id < n, fmt::format("Index out of bounds {} < {}", id, n).data());
            dc->computeDistance(data + (id * dim), result);
        }

        inline void computeDistance(vector_idx_t src, vector_idx_t dest, double *result) {
            CHECK_ARGUMENT(src < n && dest < n, "Index out of bounds");
            const T *x = data + (src * dim);
            const T *y = data + (dest * dim);
            dc->computeDistance(x, y, result);
        }

        inline void computeDistance(const T *dest, double *result) {
            dc->computeDistance(dest, result);
        }

        inline void computeDistance(const T *src, const T *dest, double *result) {
            dc->computeDistance(src, dest, result);
        }

        inline void batchComputeDistances(vector_idx_t *ids, double *results, int size) {
            for (int i = 0; i < size; i++) {
                computeDistance(ids[i], &results[i]);
            }
        }

        inline void setQuery(const float *q) {
            dc->setQuery(q);
        }

        inline std::unique_ptr<DelegateDC> clone() {
            return std::make_unique<DelegateDC>(std::unique_ptr<DistanceComputer<T>>(dc->clone()), data, dim, n);
        }

    private:
        std::unique_ptr<DistanceComputer<T>> dc;
        const T *data;
        int dim;
        int n; // number of vectors

    };

    static inline std::unique_ptr<DelegateDC<float> > createDistanceComputer(
        const float *data, int dim, int n, DistanceType distanceType) {
        std::unique_ptr<DistanceComputer<float> > delegate;
        if (distanceType == L2) {
            delegate = std::make_unique<L2DistanceComputer>(dim);
        } else if (distanceType == IP) {
            delegate = std::make_unique<IPDistanceComputer>(dim);
        } else if (distanceType == COSINE) {
            delegate = std::make_unique<CosineDistanceComputer>(dim);
        }
        return std::make_unique<DelegateDC<float> >(std::move(delegate), data, dim, n);
    }

    static inline std::unique_ptr<DelegateDC<uint8_t> > createQuantizedDistanceComputer(
        const uint8_t *data, int dim, int n, DistanceType distanceType, SQ8Bit *quantizer, bool symmetric = false) {
        std::unique_ptr<DistanceComputer<uint8_t> > delegate;
        if (symmetric) {
            if (distanceType == L2) {
                delegate = std::make_unique<SQ8SymL2Sq>(dim, quantizer);
            } else if (distanceType == IP) {
                delegate = std::make_unique<SQ8SymIP>(dim, quantizer);
            } else if (distanceType == COSINE) {
                delegate = std::make_unique<SQ8SymCosine>(dim, quantizer);
            }
        } else {
            if (distanceType == L2) {
                delegate = std::make_unique<SQ8AsymL2Sq>(dim, quantizer->getAlpha(), quantizer->getBeta(),
                                                         quantizer->getAlphaSqr());
            } else if (distanceType == IP) {
                delegate = std::make_unique<SQ8AsymIP>(dim, quantizer->getAlpha(), quantizer->getBeta(),
                                                       quantizer->getAlphaSqr());
            } else if (distanceType == COSINE) {
                delegate = std::make_unique<SQ8AsymCosine>(dim, quantizer->getAlpha(), quantizer->getBeta(),
                                                           quantizer->getAlphaSqr());
            }
        }
        return std::make_unique<DelegateDC<uint8_t> >(std::move(delegate), data, quantizer->codeSize, n);
    }

        // struct QuantizedDistanceComputer : public DistanceComputer {
        //     explicit QuantizedDistanceComputer(
        //             const uint8_t *data,
        //             fastq::DistanceComputer<float, uint8_t> *asym_dc,
        //             fastq::DistanceComputer<uint8_t, uint8_t> *sym_dc,
        //             int codeSize)
        //             : data(data), asym_dc(asym_dc), sym_dc(sym_dc), codeSize(codeSize), query(nullptr) {}
        //
        //     inline void computeDistance(vector_idx_t id, double *result) override {
        //         const uint8_t *ci = data + (id * codeSize);
        //         asym_dc->compute_distance(query, ci, result);
        //     }
        //
        //     inline void computeDistance(vector_idx_t src, vector_idx_t dest, double *result) override {
        //         const uint8_t *ci = data + (src * codeSize);
        //         const uint8_t *cj = data + (dest * codeSize);
        //         sym_dc->compute_distance(ci, cj, result);
        //     }
        //
        //     inline void computeDistance(vector_idx_t src, vector_idx_t dest, int dim, double *result) override {
        //         const uint8_t *ci = data + (src * codeSize);
        //         const uint8_t *cj = data + (dest * codeSize);
        //         sym_dc->compute_distance(ci, cj, result);
        //     }
        //
        //     inline void computeDistance(float *src, float *dest, double *result) override {
        //         // Not implemented
        //         CHECK_ARGUMENT(false, "Not implemented");
        //     }
        //
        //     inline void batchComputeDistances(vector_idx_t *ids, double *results, int size) override {
        //         for (int i = 0; i < size; i++) {
        //             computeDistance(ids[i], &results[i]);
        //         }
        //     }
        //
        //     inline void setQuery(const float *q) override {
        //         this->query = q;
        //     }
        //
        //     inline std::unique_ptr<DistanceComputer> clone() override {
        //         return std::make_unique<QuantizedDistanceComputer>(data, asym_dc, sym_dc, codeSize);
        //     }
        //
        // private:
        //     const uint8_t *data;
        //     fastq::DistanceComputer<float, uint8_t> *asym_dc;
        //     fastq::DistanceComputer<uint8_t, uint8_t> *sym_dc;
        //     int codeSize;
        //     const float *query;
        // };
        //
        // struct L2DistanceComputer : public DistanceComputer {
        //     explicit L2DistanceComputer(const float *data, int dim, int n) : data(data), dim(dim), n(n), query(nullptr) {}
        //
        //     inline void computeDistance(vector_idx_t id, double *result) override {
        //         CHECK_ARGUMENT(id < n, fmt::format("Index out of bounds {} < {}", id, n).data());
        //         const float *y = data + (id * dim);
        //         simsimd_l2sq_f32(query, y, dim, result);
        //     }
        //
        //     inline void computeDistance(vector_idx_t src, vector_idx_t dest, double *result) override {
        //         CHECK_ARGUMENT(src < n && dest < n, "Index out of bounds");
        //         const float *x = data + (src * dim);
        //         const float *y = data + (dest * dim);
        //         simsimd_l2sq_f32(x, y, dim, result);
        //     }
        //
        //     inline void computeDistance(vector_idx_t src, vector_idx_t dest, int dim, double *result) override {
        //         CHECK_ARGUMENT(src < n && dest < n, "Index out of bounds");
        //         CHECK_ARGUMENT(dim > 0 && dim <= this->dim, "Invalid dimension");
        //         const float *x = data + (src * dim);
        //         const float *y = data + (dest * dim);
        //         simsimd_l2sq_f32(x, y, dim, result);
        //     }
        //
        //     void computeDistance(float *src, float *dest, double *result) override {
        //         simsimd_l2sq_f32(src, dest, dim, result);
        //     }
        //
        //     inline void batchComputeDistances(vector_idx_t *ids, double *results, int size) override {
        //         for (int i = 0; i < size; i++) {
        //             computeDistance(ids[i], &results[i]);
        //         }
        //     }
        //
        //     void setQuery(const float *q) override {
        //         this->query = q;
        //     }
        //
        //     inline std::unique_ptr<DistanceComputer> clone() override {
        //         return std::make_unique<L2DistanceComputer>(data, dim, n);
        //     }
        //
        // private:
        //     const float *data;
        //     int dim;
        //     int n;
        //
        //     const float *query;
        // };
        //
        // struct CosineDistanceComputer : public DistanceComputer {
        //     explicit CosineDistanceComputer(const float *data, int dim, int n) : data(data), dim(dim), n(n),
        //                                                                          query(nullptr) {}
        //
        //     inline void computeDistance(vector_idx_t id, double *result) override {
        //         CHECK_ARGUMENT(id < n, fmt::format("Index out of bounds {} < {}", id, n).data());
        //         assert(id < n);
        //         const float *y = data + (id * dim);
        //         simsimd_cos_f32(query, y, dim, result);
        //     }
        //
        //     inline void computeDistance(vector_idx_t src, vector_idx_t dest, double *result) override {
        //         CHECK_ARGUMENT(src < n && dest < n, "Index out of bounds");
        //         const float *x = data + (src * dim);
        //         const float *y = data + (dest * dim);
        //         simsimd_cos_f32(x, y, dim, result);
        //     }
        //
        //     inline void computeDistance(vector_idx_t src, vector_idx_t dest, int dim, double *result) override {
        //         CHECK_ARGUMENT(src < n && dest < n, "Index out of bounds");
        //         CHECK_ARGUMENT(dim > 0 && dim <= this->dim, "Invalid dimension");
        //         const float *x = data + (src * dim);
        //         const float *y = data + (dest * dim);
        //         simsimd_cos_f32(x, y, dim, result);
        //     }
        //
        //     void computeDistance(float *src, float *dest, double *result) override {
        //         simsimd_cos_f32(src, dest, dim, result);
        //     }
        //
        //     inline void batchComputeDistances(vector_idx_t *ids, double *results, int size) override {
        //         for (int i = 0; i < size; i++) {
        //             computeDistance(ids[i], &results[i]);
        //         }
        //     }
        //
        //     inline void setQuery(const float *q) override {
        //         this->query = q;
        //     }
        //
        //     inline std::unique_ptr<DistanceComputer> clone() override {
        //         return std::make_unique<CosineDistanceComputer>(data, dim, n);
        //     }
        //
        // private:
        //     const float *data;
        //     int dim;
        //     int n;
        //
        //     const float *query;
        // };
    } // namespace orangedb
