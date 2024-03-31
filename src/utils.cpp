#include <sys/stat.h>
#include <cmath>
#include <random>
#include <cassert>
#include "include/utils.h"
#include <cstring>
#include <sys/fcntl.h>
#include <unistd.h>
#include <macros.h>

//#if defined(__GNUC__)
//#define CACHE_ALIGNED __attribute__((aligned(4))) // clang and GCC
//#elif defined(_MSC_VER)
//#define CACHE_ALIGNED __declspec(align(64))        // MSVC
//#endif
//
//typedef CACHE_ALIGNED float float_aligned_t;

namespace orangedb {
    float *Utils::fvecs_read(const char *fname, size_t *d_out, size_t *n_out) {
        FILE *f = fopen(fname, "r");
        if (!f) {
            fprintf(stderr, "could not open %s\n", fname);
            perror("");
            abort();
        }
        int d;
        fread(&d, 1, sizeof(int), f);
        assert((d > 0 && d < 1000000) || !"unreasonable dimension");
        fseek(f, 0, SEEK_SET);
        struct stat st;
        fstat(fileno(f), &st);
        size_t sz = st.st_size;
        assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
        size_t n = sz / ((d + 1) * 4);
        *d_out = d;
        *n_out = n;
        float* x = new float[n * (d + 1)];
        printf("x: %p\n", x);
        size_t nr = fread(x, sizeof(float), n * (d + 1), f);
        assert(nr == n * (d + 1) || !"could not read whole file");

        // TODO: Round up the dimensions to the nearest multiple of 8, otherwise the below code will not work
        float* align_x;
        alloc_aligned(((void**)&align_x), n * d * sizeof(float), 8 * sizeof(float));
        printf("align_x: %p\n", align_x);

        // copy data to aligned memory
        for (size_t i = 0; i < n; i++) {
            memcpy(align_x + i * d, x + 1 + i * (d + 1), d * sizeof(float));
        }

        // free original memory
        delete[] x;
        fclose(f);
        return align_x;
    }

    int* Utils::ivecs_read(const char *fname, size_t *d_out, size_t *n_out) {
        return (int*)fvecs_read(fname, d_out, n_out);
    }

    void Utils::alloc_aligned(void **ptr, size_t size, size_t align) {
        *ptr = nullptr;
        if (!IS_ALIGNED(size, align)) {
            printf("size: %lu, align: %lu\n", size, align);
            throw;
        }
#ifdef __APPLE__
        int err = posix_memalign(ptr, align, size);
        if (err)
        {
            printf("posix_memalign failed with error code %d\n", err);
            throw;
        }
#else
        *ptr = ::aligned_alloc(align, size);
#endif
        if (*ptr == nullptr) {
            printf("aligned_alloc failed\n");
            throw;
        }
    }
}
