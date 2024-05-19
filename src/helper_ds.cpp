#include "include/helper_ds.h"

namespace orangedb {
    static inline bool compareWithId(
            const vector_idx_t &id1, const float &val1, const vector_idx_t &id2, const float &val2) {
        // This is needed because if the distance is same then we need to remove element based on id. Basically
        // those which has INVALID_VECTOR_ID as id since they are already removed from heap.
        return val1 < val2 || (val1 == val2 && id1 > id2);
    }

    MaxHeap::MaxHeap(int capacity) : capacity{capacity}, physical_size{0}, logical_size{0} {
        values.resize(capacity + 1);
        ids.resize(capacity + 1);
    }

    void MaxHeap::push(vector_idx_t id, float val) {
        if (physical_size == capacity) {
            if (val >= values[1]) {
                return;
            }
            if (ids[1] == INVALID_VECTOR_ID) {
                logical_size--;
            }
            popFromHeap();
            physical_size--;
        }
        physical_size++;
        logical_size++;
        pushToHeap(id, val);
    }

    inline void MaxHeap::pushToHeap(vector_idx_t id, float val) {
        // Use 1-based indexing for easier node->child translation
        size_t i = physical_size, i_father;
        while (i > 1) {
            i_father = i >> 1;
            if (compareWithId(id, val, ids[i_father], values[i_father])) {
                // heap structure is satisfied
                break;
            }
            values[i] = values[i_father];
            ids[i] = ids[i_father];
            i = i_father;
        }
        values[i] = val;
        ids[i] = id;
    }

    void MaxHeap::popFromHeap() {
        float val = values[physical_size];
        vector_idx_t id = ids[physical_size];
        size_t i = 1, i1, i2;
        while (true) {
            i1 = i << 1;
            i2 = i1 + 1;
            if (i1 > physical_size) {
                break;
            }
            if ((i2 == physical_size + 1) || compareWithId(ids[i1], values[i1], ids[i2], values[i2])) {
                if (compareWithId(id, val, ids[i1], values[i1])) {
                    break;
                }
                values[i] = values[i1];
                ids[i] = ids[i1];
                i = i1;
            } else {
                if (compareWithId(id, val, ids[i2], values[i2])) {
                    break;
                }
                values[i] = values[i2];
                ids[i] = ids[i2];
                i = i2;
            }
        }
        values[i] = values[physical_size];
        ids[i] = ids[physical_size];
    }

    inline vector_idx_t MaxHeap::popMin(float *val) {
        // TODO: Pop min can be implemented through simd
        int i = physical_size;
        while (i > 0) {
            if (ids[i] != INVALID_VECTOR_ID) {
                break;
            }
            i--;
        }
        if (i == 0) {
            return INVALID_VECTOR_ID;
        }
        int iMin = i;
        float vMin = values[i];
        i--;
        while (i > 0) {
            if (ids[i] != INVALID_VECTOR_ID && values[i] < vMin) {
                vMin = values[i];
                iMin = i;
            }
            i--;
        }
        *val = values[iMin];
        vector_idx_t id = ids[iMin];
        ids[iMin] = INVALID_VECTOR_ID;
        logical_size--;
        return id;
    }
} // namespace orangedb
