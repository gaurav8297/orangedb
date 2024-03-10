#include <sys/stat.h>
#include <cmath>
#include <random>
#include <cassert>
#include "include/utils.h"
#include <cstring>
#include <sys/fcntl.h>
#include <unistd.h>

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
        float *x = new float[n * (d + 1)];
        size_t nr = fread(x, sizeof(float), n * (d + 1), f);
        assert(nr == n * (d + 1) || !"could not read whole file");

        // shift array to remove row headers
        for (size_t i = 0; i < n; i++)
            memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

        fclose(f);
        return x;
    }

    int* Utils::ivecs_read(const char *fname, size_t *d_out, size_t *n_out) {
        return (int*)fvecs_read(fname, d_out, n_out);
    }
}
