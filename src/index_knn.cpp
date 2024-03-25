#include "include/index_knn.h"
#include "include/distance.h"

namespace orangedb {
    void IndexKnn::search(int n, const float *queries, int k, float* distances, orangedb::storage_idx_t* result_ids) {
        assert(k == 1);
#pragma omp parallel
        {
            L2DistanceComputer dc = L2DistanceComputer(storage);
#pragma omp for
            for (int i = 0; i < n; i++) {
                dc.set_query(queries + i * storage->dim, 0);

                float min_distance = 0;
                auto j = 0, min_id = 0;
                while (j + 4 < storage->num_points) {
                    float d[4];
                    dc.compute_distance_four_vecs(j, j + 1, j + 2, j + 3, d[0], d[1], d[2], d[3]);
                    for (int l = 0; l < 4; l++) {
                        if (d[l] < min_distance) {
                            min_distance = d[l];
                            min_id = j + l;
                        }
                    }
                    j += 4;
                }

                for (int l = j; l < storage->num_points; l++) {
                    float d;
                    dc.compute_distance(l, d);
                    if (d < min_distance) {
                        min_distance = d;
                        min_id = l;
                    }
                }
                distances[i] = min_distance;
                result_ids[i] = min_id;
            }
        }
    }
}
