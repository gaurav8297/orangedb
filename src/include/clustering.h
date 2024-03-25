#pragma once

namespace orangedb {
    class Clustering {
    public:
        Clustering(int d, int k);
        void train(
                int n,
                const float* x,
                const float* x_weights = nullptr);

        void train_encoded(
                int n,
                const uint8_t* x_in,
                const float* weights);

        int subsample_training_set(
                int nx,
               const uint8_t* x,
               size_t line_size,
               const float* weights,
               uint8_t** x_out,
               float** weights_out);

        inline void rand_perm(int* perm, size_t n, int64_t seed) {
//            for (size_t i = 0; i < n; i++)
//                perm[i] = i;
//
//            for (size_t i = 0; i + 1 < n; i++) {
//                int i2 = i + rng.rand_int(n - i);
//                std::swap(perm[i], perm[i2]);
//            }
        }

    private:
        int d; // dimension
        int k; // number of centroids

        int n_redo; // number of redo
        int n_iter; // number of iterations

        std::vector<float> centroids;
    };
}
