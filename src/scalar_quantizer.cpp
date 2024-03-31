#include "include/scalar_quantizer.h"

namespace orangedb {
    void ScalarQuantizer::train(size_t n, const float *x) {
        trained.resize(2 * d);
        float* vmin = trained.data();
        float* vmax = trained.data() + d;
        memcpy(vmin, x, sizeof(*x) * d);
        memcpy(vmax, x, sizeof(*x) * d);
        for (int i = 1; i < n; i++) {
            const float* xi = x + i * d;
            for (int j = 0; j < d; j++) {
                vmin[j] = std::min(vmin[j], xi[j]);
                vmax[j] = std::max(vmax[j], xi[j]);
            }
        }
        float* vdiff = vmax;
        for (size_t j = 0; j < d; j++) {
            float vexp = (vmax[j] - vmin[j]) * rangestat_arg;
            // Expand the range by a factor rangestat_arg
            vmin[j] -= vexp;
            vmax[j] += vexp;
            vdiff[j] = vmax[j] - vmin[j];
        }
    }

    void ScalarQuantizer::compute_codes(const float *x, uint8_t *codes, size_t n) const {
        for (size_t i = 0; i < n; i++) {
            const float* xi = x + i * d;
            uint8_t* ci = codes + i * d;
            for (size_t j = 0; j < d; j++) {
                float xij = xi[j];
                float vmin = trained[j];
                float vdiff = trained[j + d];
                // Scale to [0, 1]
                float descaled_val = (xij - vmin) / vdiff;
                // Scale to [0, 255]
                ci[j] = int(descaled_val * 255.0f);
            }
        }
    }

    void ScalarQuantizer::decode(const uint8_t *code, float *x, size_t n) const {
        for (size_t i = 0; i < n; i++) {
            const uint8_t* ci = code + i * d;
            float* xi = x + i * d;
            for (size_t j = 0; j < d; j++) {
                float vmin = trained[j];
                float vdiff = trained[j + d];

                // Scale to [0, 1]. Reason to add 0.5f is to round the value.
                // This is used for continuity correction. Basically the probability that a random variable
                // falls in a certain range is the same as the probability that the random variable falls in the
                // corresponding integer value. It gives out a more accurate result.
                float scaled_val = (ci[j] + 0.5f) / 255.0f;
                xi[j] = vmin + scaled_val * vdiff;
            }
        }
    }

#ifdef __AVX2__

#endif

    void ScalarQuantizer::print_stats() {
        for (size_t i = 0; i < d; i++) {
            printf("vmin[%ld] = %f, vmax[%ld] = %f\n", i, trained[i], i, trained[i + d]);
        }
    }
} // namespace orangedb
