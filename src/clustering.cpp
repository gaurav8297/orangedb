//
// Created by Gaurav Sehgal on 2024-03-24.
//

#include <cassert>
#include <memory>
#include <vector>
#include "include/clustering.h"
#include <cstring>

#define MAX_CENTROID_SIZE 256

namespace orangedb {
    Clustering::Clustering(int d, int k) {
        this->d = d;
        this->k = k;
    }

    void Clustering::train(int n, const float *x, const float *x_weights) {
        train_encoded(n, reinterpret_cast<const uint8_t*>(x), x_weights);
    }

    void Clustering::train_encoded(int n, const uint8_t *x_in, const float *weights_in) {
        // TODO - implement training
        // Number of training vectors should be greater than number of centroids
        assert(n > k);

        uint8_t* x = nullptr;
        float *weights = nullptr;
        size_t line_size = sizeof(float) * d;

        // Assuming n is greater than k * MAX_CENTROID_SIZE
        int nx = subsample_training_set(n, x_in, line_size, weights_in, &x, &weights);
        std::unique_ptr<uint32_t[]> assign(new uint32_t[nx]);
        std::unique_ptr<float[]> dis(new float[nx]);

        // Initialize centroids
        std::vector<float> best_centroids;

        for (int redo = 0; redo < n_redo; redo++) {
            centroids.resize(d * k);
            std::vector<int> perm(nx);
            rand_perm(perm.data(), nx, 1235 + redo * 15486557L);

            for (int i = 0; i < k; i++) {
                memcpy(&centroids[i * d], x + perm[i] * line_size, line_size);
            }

            // K-means iterations
            for (int i = 0; i < n_iter; i++) {

            }
        }
    }

    int Clustering::subsample_training_set(int nx, const uint8_t *x, size_t line_size, const float *weights,
                                           uint8_t **x_out, float **weights_out) {
        std::vector<int> perm(nx);
        // For every i in perm, it should be greater or equal to at-least i, otherwise we will copy the same vector
        rand_perm(perm.data(), nx, 1234);
        nx = k * MAX_CENTROID_SIZE;
        uint8_t* x_new = new uint8_t[nx * line_size];
        *x_out = x_new;
        for (int i = 0; i < nx; i++) {
            memcpy(x_new + i * line_size, x + perm[i] * line_size, line_size);
        }
        *weights_out = nullptr;
        return nx;
    }
}
