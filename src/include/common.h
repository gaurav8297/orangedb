#pragma once

#include <sys/stat.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <sys/fcntl.h>
#include <limits>
#include <random>
#include <vector>
#include <faiss/utils/random.h>
#include "spdlog/fmt/fmt.h"
#include <unordered_map>
#include <simsimd/simsimd.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)
#define IS_512_ALIGNED(X) IS_ALIGNED(X, 512)

namespace orangedb {
    typedef uint8_t level_t;
    typedef uint64_t vector_idx_t;
    constexpr vector_idx_t INVALID_VECTOR_ID = UINT64_MAX;

    [[noreturn]] inline void failCheckArgument(
        const char *condition_name, const char *file, int linenr, const char *comment) {
        throw std::invalid_argument(fmt::format(
            "Assertion failed in file \"{}\" on line {}: {} with comment: {}", file, linenr, condition_name,
            comment));
    }

#define CHECK_ARGUMENT(condition, comment)                                                            \
    static_cast<bool>(condition) ?                                                                    \
        void(0) :                                                                                     \
        orangedb::failCheckArgument(#condition, __FILE__, __LINE__, static_cast<const char*>(comment))

    static void allocAligned(void **ptr, size_t size, size_t align) {
        *ptr = nullptr;
        if (!IS_ALIGNED(size, align)) {
            printf("size: %lu, align: %lu\n", size, align);
            throw;
        }
#ifdef __APPLE__
        int err = posix_memalign(ptr, align, size);
        if (err) {
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

    static float *readFvecFile(const char *fName, size_t *d_out, size_t *n_out) {
        FILE *f = fopen(fName, "r");
        if (!f) {
            fprintf(stderr, "could not open %s\n", fName);
            perror("");
            abort();
        }
        int d;
        fread(&d, 1, sizeof(int), f);
        CHECK_ARGUMENT((d > 0 && d < 1000000), "unreasonable dimension");
        fseek(f, 0, SEEK_SET);
        struct stat st{};
        fstat(fileno(f), &st);
        size_t sz = st.st_size;
        CHECK_ARGUMENT(sz % ((d + 1) * 4) == 0, "weird file size");
        size_t n = sz / ((d + 1) * 4);
        *d_out = d;
        *n_out = n;
        auto *x = new float[n * (d + 1)];
        printf("x: %p\n", x);
        size_t nr = fread(x, sizeof(float), n * (d + 1), f);
        CHECK_ARGUMENT(nr == n * (d + 1), "could not read whole file");

        // TODO: Round up the dimensions to the nearest multiple of 8, otherwise the below code will not work
        float *align_x;
        allocAligned(((void **) &align_x), n * d * sizeof(float), 8 * sizeof(float));
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

    static void readBvecFileStats(const char *fName, size_t *d_out, size_t *n_out);

    static float *readBvecFileChunk(
            const char *fName,
            size_t start_row,
            size_t max_rows,
            size_t *d_out,
            size_t *n_out);

    static float *readBvecTrainingSample(
            const char *fName,
            size_t sample_rows,
            size_t *d_out,
            size_t *n_out,
            double sample_percent = 0.0,
            size_t max_source_rows = SIZE_MAX,
            uint64_t seed = 1234);

    static float *readBvecFile(const char *fName, size_t *d_out, size_t *n_out, size_t max_rows = SIZE_MAX) {
        return readBvecFileChunk(fName, 0, max_rows, d_out, n_out);
    }

    static void readBvecFileStats(const char *fName, size_t *d_out, size_t *n_out) {
        FILE *f = fopen(fName, "rb");
        if (!f) {
            fprintf(stderr, "could not open %s\n", fName);
            perror("");
            abort();
        }

        int d;
        fread(&d, 1, sizeof(int), f);
        CHECK_ARGUMENT((d > 0 && d < 1000000), "unreasonable dimension");

        struct stat st{};
        fstat(fileno(f), &st);
        size_t sz = st.st_size;
        const size_t bytes_per_vector = 4 + static_cast<size_t>(d) * sizeof(uint8_t);
        CHECK_ARGUMENT(sz % bytes_per_vector == 0, "weird file size");

        *d_out = d;
        *n_out = sz / bytes_per_vector;
        fclose(f);
    }

    static float *readBvecFileChunk(
            const char *fName,
            size_t start_row,
            size_t max_rows,
            size_t *d_out,
            size_t *n_out) {
        size_t d;
        size_t total_n;
        readBvecFileStats(fName, &d, &total_n);
        CHECK_ARGUMENT(start_row <= total_n, "start row exceeds total rows");

        const size_t n = std::min(max_rows, total_n - start_row);
        *d_out = d;
        *n_out = n;
        if (n == 0) {
            return nullptr;
        }

        FILE *f = fopen(fName, "rb");
        if (!f) {
            fprintf(stderr, "could not open %s\n", fName);
            perror("");
            abort();
        }

        const size_t bytes_per_vector = 4 + d * sizeof(uint8_t);
        const size_t offset = start_row * bytes_per_vector;
        CHECK_ARGUMENT(fseek(f, static_cast<long>(offset), SEEK_SET) == 0, "failed to seek in bvec file");

        auto *x = new uint8_t[n * bytes_per_vector];
        printf("x: %p\n", x);
        size_t nr = fread(x, sizeof(uint8_t), n * bytes_per_vector, f);
        CHECK_ARGUMENT(nr == n * bytes_per_vector, "could not read whole chunk");

        float *align_x;
        allocAligned((void **) &align_x, n * d * sizeof(float), 8 * sizeof(float));
        printf("align_x: %p\n", align_x);

        for (size_t i = 0; i < n; i++) {
            const uint8_t *row = x + i * bytes_per_vector + 4;
            for (size_t j = 0; j < d; j++) {
                align_x[i * d + j] = static_cast<float>(row[j]);
            }
        }

        delete[] x;
        fclose(f);
        return align_x;
    }

    static size_t resolveBvecSampleCount(size_t total_rows, size_t sample_rows, double sample_percent) {
        if (sample_percent > 0.0) {
            auto resolved = static_cast<size_t>(std::ceil((sample_percent / 100.0) * total_rows));
            return std::min(total_rows, std::max<size_t>(1, resolved));
        }
        return std::min(total_rows, sample_rows);
    }

    static float *readBvecTrainingSample(
            const char *fName,
            size_t sample_rows,
            size_t *d_out,
            size_t *n_out,
            double sample_percent,
            size_t max_source_rows,
            uint64_t seed) {
        size_t d = 0;
        size_t total_rows = 0;
        readBvecFileStats(fName, &d, &total_rows);
        total_rows = std::min(total_rows, max_source_rows);

        const size_t resolved_sample_rows = resolveBvecSampleCount(total_rows, sample_rows, sample_percent);
        *d_out = d;
        *n_out = resolved_sample_rows;

        if (resolved_sample_rows == 0 || total_rows == 0) {
            return nullptr;
        }
        if (resolved_sample_rows == total_rows) {
            return readBvecFileChunk(fName, 0, total_rows, d_out, n_out);
        }

        FILE *f = fopen(fName, "rb");
        if (!f) {
            fprintf(stderr, "could not open %s\n", fName);
            perror("");
            abort();
        }

        const size_t bytes_per_vector = 4 + d * sizeof(uint8_t);
        std::vector<uint8_t> row(bytes_per_vector);
        CHECK_ARGUMENT(
                total_rows <= static_cast<size_t>(std::numeric_limits<int>::max()),
                "faiss::rand_perm requires total_rows to fit in int");

        struct SampleRowRef {
            size_t row_idx;
            size_t sample_idx;
        };

        std::vector<int> perm(total_rows);
        faiss::rand_perm(perm.data(), total_rows, static_cast<int64_t>(seed));

        std::vector<SampleRowRef> sampled_rows;
        sampled_rows.reserve(resolved_sample_rows);
        for (size_t sample_idx = 0; sample_idx < resolved_sample_rows; sample_idx++) {
            sampled_rows.push_back(
                    SampleRowRef{static_cast<size_t>(perm[sample_idx]), sample_idx});
        }
        std::sort(
                sampled_rows.begin(),
                sampled_rows.end(),
                [](const SampleRowRef &lhs, const SampleRowRef &rhs) { return lhs.row_idx < rhs.row_idx; });

        float *sampled_vecs;
        allocAligned((void **) &sampled_vecs, resolved_sample_rows * d * sizeof(float), 8 * sizeof(float));

        auto copy_row = [&](size_t sample_idx) {
            const uint8_t *src = row.data() + 4;
            float *dst = sampled_vecs + sample_idx * d;
            for (size_t j = 0; j < d; j++) {
                dst[j] = static_cast<float>(src[j]);
            }
        };

        for (const auto &sampled_row : sampled_rows) {
            const off_t offset = static_cast<off_t>(sampled_row.row_idx * bytes_per_vector);
            CHECK_ARGUMENT(fseeko(f, offset, SEEK_SET) == 0, "failed to seek in bvec file");
            size_t nr = fread(row.data(), sizeof(uint8_t), bytes_per_vector, f);
            CHECK_ARGUMENT(nr == bytes_per_vector, "could not read whole vector");
            copy_row(sampled_row.sample_idx);
        }

        fclose(f);
        return sampled_vecs;
    }

    static void writeFvecFile(const char *fName, const float *data, size_t d, size_t n) {
        // Open the file in binary write mode
        FILE *f = fopen(fName, "wb");
        if (!f) {
            fprintf(stderr, "could not open %s for writing\n", fName);
            perror("");
            abort();
        }

        // Allocate a temporary buffer for storing dimension + float values
        auto *buffer = new float[(d + 1) * n]; // dimension (1 float/4 bytes) + d floats per vector

        // Fill the buffer with dimension + vector data
        for (size_t i = 0; i < n; i++) {
            // Store the dimension as an int (reinterpreted as float storage)
            int dimension = static_cast<int>(d);
            memcpy(buffer + i * (d + 1), &dimension, sizeof(int));

            // Copy the float vector data
            memcpy(buffer + i * (d + 1) + 1, data + i * d, d * sizeof(float));
        }

        // Write the buffer to the file
        size_t elements_written = fwrite(buffer, sizeof(float), n * (d + 1), f);
        CHECK_ARGUMENT(elements_written == n * (d + 1), "could not write whole file");

        // Free the buffer and close the file
        delete[] buffer;
        fclose(f);
    }

    static void writeBvecFile(const char *fName, const float *data, size_t d, size_t n) {
        // Open the file in binary write mode
        FILE *f = fopen(fName, "wb");
        if (!f) {
            fprintf(stderr, "could not open %s for writing\n", fName);
            perror("");
            abort();
        }

        // Allocate a temporary buffer for storing uint8_t values
        auto *buffer = new uint8_t[(d + 4) * n]; // 4 bytes for dimension + d bytes for vector values per vector

        // Fill the buffer with dimension + vector data
        for (size_t i = 0; i < n; i++) {
            // Store the dimension (4 bytes)
            int dimension = static_cast<int>(d);
            memcpy(&buffer[i * (d + 4)], &dimension, sizeof(int));

            // Convert float data to uint8_t and store in buffer
            for (size_t j = 0; j < d; j++) {
                // Convert float to uint8_t (clamp to [0, 255] range)
                float value = data[i * d + j];
                buffer[i * (d + 4) + 4 + j] = static_cast<uint8_t>(value < 0.0f ? 0 : (value > 255.0f ? 255 : value));
            }
        }

        // Write the buffer to the file
        size_t bytes_written = fwrite(buffer, sizeof(uint8_t), n * (d + 4), f);
        CHECK_ARGUMENT(bytes_written == n * (d + 4), "could not write whole file");

        // Free the buffer and close the file
        delete[] buffer;
        fclose(f);
    }

    static float *readVecFile(const char *fName, size_t *d_out, size_t *n_out, size_t max_rows = SIZE_MAX) {
        bool is_bvecs = false;
        if (strstr(fName, ".bvecs")) {
            is_bvecs = true;
        }
        if (is_bvecs) {
            return readBvecFile(fName, d_out, n_out, max_rows);
        } else {
            return readFvecFile(fName, d_out, n_out);
        }
    }

    static int *readIvecFile(const char *fName, size_t *d_out, size_t *n_out) {
        return (int *) readFvecFile(fName, d_out, n_out);
    }

    static float *readFbinFile(const char *fName, size_t *d_out, size_t *n_out) {
        FILE *f = fopen(fName, "rb");
        if (!f) {
            fprintf(stderr, "could not open %s\n", fName);
            perror("");
            abort();
        }
        // Read num of vecs
        int n_int;
        fread(&n_int, sizeof(int), 1, f);
        // Read dimension
        int d_int;
        fread(&d_int, sizeof(int), 1, f);
        *d_out = d_int;
        *n_out = n_int;

        auto *x = new float[n_int * d_int];
        size_t bytes_read = fread(x, sizeof(float), n_int * d_int, f);
        CHECK_ARGUMENT(bytes_read == n_int * d_int, "could not read whole file");

        fclose(f);
        return x;
    }

    static void writeFbinFile(const char *fName, const float *data, size_t d, size_t n) {
        FILE *f = fopen(fName, "wb");
        if (!f) {
            fprintf(stderr, "could not open %s for writing\n", fName);
            perror("");
            abort();
        }
        // Write num of vecs
        int n_int = static_cast<int>(n);
        fwrite(&n_int, sizeof(int), 1, f);
        // Write dimension
        int d_int = static_cast<int>(d);
        fwrite(&d_int, sizeof(int), 1, f);
        // Write data
        size_t bytes_written = fwrite(data, sizeof(float), n * d, f);
        CHECK_ARGUMENT(bytes_written == n * d, "could not write whole file");

        // Close the file
        fclose(f);
    }

    static void list_parquet_dir(const char *dir_path, std::vector<std::string> &file_paths) {
        try {
            for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
                if (entry.is_regular_file() && entry.path().extension() == ".parquet") {
                    file_paths.emplace_back(entry.path().string());
                }
            }
            std::sort(file_paths.begin(), file_paths.end());
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Filesystem error: " << e.what() << '\n';
        } catch (const std::exception& e) {
            std::cerr << "General error: " << e.what() << '\n';
        }
    }

    static constexpr int64_t PARQUET_EMB_BATCH_SIZE = 65536;

    static arrow::Result<std::unique_ptr<parquet::arrow::FileReader>> openParquetArrowReader(const char *fName) {
        ARROW_ASSIGN_OR_RAISE(auto infile, arrow::io::ReadableFile::Open(std::string(fName)));
        ARROW_ASSIGN_OR_RAISE(auto reader, parquet::arrow::OpenFile(infile, arrow::default_memory_pool()));
        reader->set_use_threads(false);
        return reader;
    }

    static arrow::Result<int> getParquetEmbeddingColumnIndex(
            parquet::arrow::FileReader *reader,
            std::shared_ptr<arrow::Schema> *schema_out,
            const std::string &column_name = "emb") {
        std::shared_ptr<arrow::Schema> schema;
        ARROW_RETURN_NOT_OK(reader->GetSchema(&schema));
        const int col_index = schema->GetFieldIndex(column_name);
        if (col_index == -1) {
            return arrow::Status::Invalid("Column '" + column_name + "' not found");
        }
        *schema_out = std::move(schema);
        return col_index;
    }

    static arrow::Result<size_t> inferParquetEmbeddingDimFromArray(
            const std::shared_ptr<arrow::Array> &array,
            const std::string &column_name = "emb") {
        switch (array->type_id()) {
            case arrow::Type::FIXED_SIZE_LIST: {
                const auto fixed_list = std::static_pointer_cast<arrow::FixedSizeListArray>(array);
                const auto float_values = std::dynamic_pointer_cast<arrow::FloatArray>(fixed_list->values());
                if (!float_values) {
                    return arrow::Status::Invalid("Column '" + column_name + "' values must be float32");
                }
                return static_cast<size_t>(fixed_list->value_length());
            }
            case arrow::Type::LIST: {
                const auto list = std::static_pointer_cast<arrow::ListArray>(array);
                const auto float_values = std::dynamic_pointer_cast<arrow::FloatArray>(list->values());
                if (!float_values) {
                    return arrow::Status::Invalid("Column '" + column_name + "' values must be float32");
                }
                if (list->length() == 0) {
                    return arrow::Status::Invalid("Cannot infer dimension from empty list batch");
                }
                return static_cast<size_t>(list->value_length(0));
            }
            case arrow::Type::LARGE_LIST: {
                const auto list = std::static_pointer_cast<arrow::LargeListArray>(array);
                const auto float_values = std::dynamic_pointer_cast<arrow::FloatArray>(list->values());
                if (!float_values) {
                    return arrow::Status::Invalid("Column '" + column_name + "' values must be float32");
                }
                if (list->length() == 0) {
                    return arrow::Status::Invalid("Cannot infer dimension from empty large-list batch");
                }
                return static_cast<size_t>(list->value_length(0));
            }
            default:
                return arrow::Status::Invalid(
                        "Column '" + column_name +
                        "' must be FixedSizeList<float>, List<float>, or LargeList<float>");
        }
    }

    static arrow::Result<size_t> inferParquetEmbeddingDim(
            parquet::arrow::FileReader *reader,
            const std::shared_ptr<arrow::Schema> &schema,
            int col_index,
            const std::string &column_name = "emb") {
        const auto field = schema->field(col_index);
        if (field->type()->id() == arrow::Type::FIXED_SIZE_LIST) {
            const auto fixed_list = std::static_pointer_cast<arrow::FixedSizeListType>(field->type());
            if (fixed_list->value_type()->id() != arrow::Type::FLOAT) {
                return arrow::Status::Invalid("Column '" + column_name + "' values must be float32");
            }
            return static_cast<size_t>(fixed_list->list_size());
        }

        std::vector<int> row_group_indices(reader->num_row_groups());
        std::iota(row_group_indices.begin(), row_group_indices.end(), 0);
        reader->set_batch_size(1);
        ARROW_ASSIGN_OR_RAISE(auto batch_reader, reader->GetRecordBatchReader(row_group_indices, {col_index}));
        while (true) {
            std::shared_ptr<arrow::RecordBatch> batch;
            ARROW_RETURN_NOT_OK(batch_reader->ReadNext(&batch));
            if (!batch) {
                break;
            }
            if (batch->num_rows() == 0) {
                continue;
            }
            return inferParquetEmbeddingDimFromArray(batch->column(0), column_name);
        }

        return arrow::Status::Invalid("Column '" + column_name + "' contains no rows");
    }

    static arrow::Status copyParquetEmbeddingBatch(
            const std::shared_ptr<arrow::Array> &array,
            float *output,
            size_t expected_dim,
            size_t *rows_out,
            const std::string &column_name = "emb") {
        *rows_out = 0;
        CHECK_ARGUMENT(array != nullptr, "batch column array is null");
        CHECK_ARGUMENT(array->null_count() == 0, "Parquet embedding column contains null rows");

        switch (array->type_id()) {
            case arrow::Type::FIXED_SIZE_LIST: {
                const auto fixed_list = std::static_pointer_cast<arrow::FixedSizeListArray>(array);
                const auto float_values = std::dynamic_pointer_cast<arrow::FloatArray>(fixed_list->values());
                if (!float_values) {
                    return arrow::Status::Invalid("Column '" + column_name + "' values must be float32");
                }
                CHECK_ARGUMENT(float_values->null_count() == 0, "Parquet embedding column contains null float values");
                const size_t dim = static_cast<size_t>(fixed_list->value_length());
                CHECK_ARGUMENT(dim == expected_dim, "Parquet embedding dimension mismatch");
                const int64_t value_offset = fixed_list->value_offset(0);
                std::memcpy(
                        output,
                        float_values->raw_values() + value_offset,
                        static_cast<size_t>(fixed_list->length()) * dim * sizeof(float));
                *rows_out = static_cast<size_t>(fixed_list->length());
                return arrow::Status::OK();
            }
            case arrow::Type::LIST: {
                const auto list = std::static_pointer_cast<arrow::ListArray>(array);
                const auto float_values = std::dynamic_pointer_cast<arrow::FloatArray>(list->values());
                if (!float_values) {
                    return arrow::Status::Invalid("Column '" + column_name + "' values must be float32");
                }
                CHECK_ARGUMENT(float_values->null_count() == 0, "Parquet embedding column contains null float values");
                const float *src = float_values->raw_values();
                for (int64_t i = 0; i < list->length(); i++) {
                    CHECK_ARGUMENT(!list->IsNull(i), "Parquet embedding column contains null rows");
                    const size_t dim = static_cast<size_t>(list->value_length(i));
                    CHECK_ARGUMENT(dim == expected_dim, "Parquet embedding dimension mismatch");
                    std::memcpy(
                            output + static_cast<size_t>(i) * expected_dim,
                            src + list->value_offset(i),
                            expected_dim * sizeof(float));
                }
                *rows_out = static_cast<size_t>(list->length());
                return arrow::Status::OK();
            }
            case arrow::Type::LARGE_LIST: {
                const auto list = std::static_pointer_cast<arrow::LargeListArray>(array);
                const auto float_values = std::dynamic_pointer_cast<arrow::FloatArray>(list->values());
                if (!float_values) {
                    return arrow::Status::Invalid("Column '" + column_name + "' values must be float32");
                }
                CHECK_ARGUMENT(float_values->null_count() == 0, "Parquet embedding column contains null float values");
                const float *src = float_values->raw_values();
                for (int64_t i = 0; i < list->length(); i++) {
                    CHECK_ARGUMENT(!list->IsNull(i), "Parquet embedding column contains null rows");
                    const size_t dim = static_cast<size_t>(list->value_length(i));
                    CHECK_ARGUMENT(dim == expected_dim, "Parquet embedding dimension mismatch");
                    std::memcpy(
                            output + static_cast<size_t>(i) * expected_dim,
                            src + list->value_offset(i),
                            expected_dim * sizeof(float));
                }
                *rows_out = static_cast<size_t>(list->length());
                return arrow::Status::OK();
            }
            default:
                return arrow::Status::Invalid(
                        "Column '" + column_name +
                        "' must be FixedSizeList<float>, List<float>, or LargeList<float>");
        }
    }

    static arrow::Status readParquetFileStats(
            const char *fName,
            size_t *d_out,
            size_t *n_out,
            const std::string &column_name = "emb") {
        ARROW_ASSIGN_OR_RAISE(auto reader, openParquetArrowReader(fName));
        std::shared_ptr<arrow::Schema> schema;
        ARROW_ASSIGN_OR_RAISE(
                const int col_index,
                getParquetEmbeddingColumnIndex(reader.get(), &schema, column_name));
        ARROW_ASSIGN_OR_RAISE(*d_out, inferParquetEmbeddingDim(reader.get(), schema, col_index, column_name));
        std::shared_ptr<parquet::FileMetaData> meta = reader->parquet_reader()->metadata();
        *n_out = meta->num_rows();
        return arrow::Status::OK();
    }

    static arrow::Status readParquetFile(
            const char *fName,
            float* output,
            size_t expected_dim,
            size_t *n_out,
            const std::string &column_name = "emb") {
        ARROW_ASSIGN_OR_RAISE(auto reader, openParquetArrowReader(fName));
        std::shared_ptr<arrow::Schema> schema;
        ARROW_ASSIGN_OR_RAISE(
                const int col_index,
                getParquetEmbeddingColumnIndex(reader.get(), &schema, column_name));
        printf("Reading column '%s' at index %d\n", column_name.c_str(), col_index);

        std::vector<int> row_group_indices(reader->num_row_groups());
        std::iota(row_group_indices.begin(), row_group_indices.end(), 0);
        reader->set_batch_size(PARQUET_EMB_BATCH_SIZE);
        ARROW_ASSIGN_OR_RAISE(auto batch_reader, reader->GetRecordBatchReader(row_group_indices, {col_index}));

        *n_out = 0;
        while (true) {
            std::shared_ptr<arrow::RecordBatch> batch;
            ARROW_RETURN_NOT_OK(batch_reader->ReadNext(&batch));
            if (!batch) {
                break;
            }
            if (batch->num_rows() == 0) {
                continue;
            }

            size_t rows_read = 0;
            ARROW_RETURN_NOT_OK(copyParquetEmbeddingBatch(
                    batch->column(0),
                    output + (*n_out) * expected_dim,
                    expected_dim,
                    &rows_read,
                    column_name));
            *n_out += rows_read;
        }
        return arrow::Status::OK();
    }

    static float* readParquetFiles(
            const std::vector<std::string>& file_paths,
            size_t *d_out,
            size_t *n_out,
            const std::string &column_name = "emb") {
        *n_out = 0;
        for (const auto& file_path : file_paths) {
            size_t file_dim = 0;
            size_t total_rows = 0;
            if (auto res = readParquetFileStats(file_path.c_str(), &file_dim, &total_rows, column_name); !res.ok()) {
                throw std::runtime_error(
                        fmt::format("Failed to read Parquet file stats for {}: {}", file_path, res.ToString()));
            }
            if (*n_out == 0) {
                *d_out = file_dim;
            } else {
                CHECK_ARGUMENT(*d_out == file_dim, "Parquet files have inconsistent embedding dimensions");
            }
            *n_out += total_rows;
        }

        float *buffer;
        allocAligned(((void **) &buffer), *n_out * *d_out * sizeof(float), 8 * sizeof(float));
        printf("align_x: %p\n", buffer);
        printf("Total vectors to read: %zu, Dimension: %zu\n", *n_out, *d_out);

        size_t idx = 0;
        for (const auto& file_path : file_paths) {
            printf("Reading Parquet file: %s\n", file_path.c_str());
            size_t total_rows = 0;
            if (auto res = readParquetFile(
                    file_path.c_str(), buffer + idx * (*d_out), *d_out, &total_rows, column_name); !res.ok()) {
                throw std::runtime_error(
                        fmt::format("Failed to read Parquet file {}: {}", file_path, res.ToString()));
            }
            idx += total_rows;
        }
        printf("Finished reading Parquet files. Total vectors read: %zu\n", idx);
        return buffer;
    }

    static float* readParquetDir(
            const char *dir_path,
            size_t *d_out,
            size_t *n_out,
            const std::string &column_name = "emb") {
        std::vector<std::string> file_paths;
        list_parquet_dir(dir_path, file_paths);
        if (file_paths.empty()) {
            throw std::runtime_error("No Parquet files found in the directory");
        }
        return readParquetFiles(file_paths, d_out, n_out, column_name);
    }

    struct RandomGenerator {
        RandomGenerator(int seed) : mt(seed) {
        };

        inline float randFloat() {
            return mt() / float(mt.max());
        }

        inline int randInt(int max) {
            return mt() % max;
        }

        inline void randomPerm(uint64_t n, uint64_t *perm, uint64_t nPerm) {
            CHECK_ARGUMENT(nPerm <= n, "Number of permutations should be less than the number of elements");
            std::unordered_map<uint64_t, uint64_t> m;
            for (int i = 0; i < nPerm - 1; i++) {
                auto i2 = i + randInt(n - i);
                if (m.contains(i2)) {
                    perm[i] = m[i2];
                } else {
                    perm[i] = i2;
                }
                m[i2] = i;
            }

            // last element
            if (m.contains(nPerm - 1)) {
                perm[nPerm - 1] = m[nPerm - 1];
            } else {
                perm[nPerm - 1] = nPerm - 1;
            }
        }

        std::mt19937 mt;
    };


#if _SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("+simd")
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)

    inline static float compute_normalized_factor_neon(const float *vector, int dim) {
        float32x4_t sum_vec = vdupq_n_f32(0.0f); // Initialize sum vector to 0
        int i = 0;
        // Process 4 elements at a time
        for (; i + 4 <= dim; i += 4) {
            float32x4_t vec = vld1q_f32(vector + i); // Load 4 elements
            sum_vec = vfmaq_f32(sum_vec, vec, vec); // Square each element
        }
        // Horizontal addition of sum_vec components
        float sum = vaddvq_f32(sum_vec);

        // Handle the remaining elements (if dim is not divisible by 4)
        for (; i < dim; i++) {
            sum += vector[i] * vector[i];
        }

        return 1.0f / std::sqrt(sum); // Compute the normalization factor
    }

    inline static void normalize_vectors_neon(const float *vector, int dim, float *normalized_vector) {
        float norm = compute_normalized_factor_neon(vector, dim);
        float32x4_t norm_vec = vdupq_n_f32(norm); // Create a vector with the normalization factor
        int i = 0;
        // Process 4 elements at a time
        for (; i + 4 <= dim; i += 4) {
            float32x4_t vec = vld1q_f32(vector + i); // Load 4 elements
            float32x4_t normed_vec = vmulq_f32(vec, norm_vec); // Normalize the vector
            vst1q_f32(normalized_vector + i, normed_vec); // Store the normalized vector
        }

        // Handle the remaining elements (if dim is not divisible by 4)
        for (; i < dim; i++) {
            normalized_vector[i] = vector[i] * norm;
        }
    }

#pragma clang attribute pop
#pragma GCC pop_options
#endif
#endif

#if _SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "fma")
#pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to = function)

inline static float compute_normalized_factor_haswell(const float *vector, int dim) {
    __m256 sum_vec = _mm256_setzero_ps();  // Initialize sum vector to 0
    int i = 0;

    // Process 8 elements at a time using AVX2
    for (; i + 8 <= dim; i += 8) {
        __m256 vec = _mm256_loadu_ps(vector + i);
        // Use FMA for multiply-add operation: sum += vec * vec
        sum_vec = _mm256_fmadd_ps(vec, vec, sum_vec);
    }

    // Reduce with double precision for better accuracy
    double sum = _simsimd_reduce_f32x8_haswell(sum_vec);

    // Handle remaining elements in double precision
    for (; i < dim; i++) {
        sum += vector[i] * vector[i];
    }

    return 1.0 / std::sqrt(sum);
}

inline static void normalize_vectors_haswell(const float *vector, int dim, float *normalized_vector) {
    float norm = compute_normalized_factor_haswell(vector, dim);
    __m256 norm_vec = _mm256_set1_ps(norm);  // Broadcast norm to all elements
    int i = 0;

    // Process 8 elements at a time
    for (; i + 8 <= dim; i += 8) {
        __m256 vec = _mm256_loadu_ps(vector + i);
        __m256 normed_vec = _mm256_mul_ps(vec, norm_vec);
        _mm256_storeu_ps(normalized_vector + i, normed_vec);
    }

    // Handle remaining elements
    for (; i < dim; i++) {
        normalized_vector[i] = vector[i] * norm;
    }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2"))), apply_to = function)
    inline static float compute_normalized_factor_skylake(const float *vector, int dim) {
        __m512 sum_vec = _mm512_setzero_ps();  // Initialize sum vector to 0
        int i = 0;

        for (; i + 16 <= dim; i += 16) {
            __m512 vec = _mm512_loadu_ps(vector + i);     // Load 16 elements
            sum_vec= _mm512_fmadd_ps(vec, vec, sum_vec);     // Square each element
        }

        float sum = _mm512_reduce_add_ps(sum_vec);

        for (; i < dim; i++) {
            sum += vector[i] * vector[i];
        }

        return 1.0f / std::sqrt(sum); // Compute the normalization factor
    }

    inline static void normalize_vectors_skylake(const float *vector, int dim, float *normalized_vector) {
        float norm = compute_normalized_factor_skylake(vector, dim);
        __m512 norm_vec = _mm512_set1_ps(norm); // Broadcast norm to all elements
        int i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 vec = _mm512_loadu_ps(vector + i);
            __m512 normed_vec = _mm512_mul_ps(vec, norm_vec);
            _mm512_storeu_ps(normalized_vector + i, normed_vec);
        }
        for (; i < dim; i++) {
            normalized_vector[i] = vector[i] * norm;
        }
    }
#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SKYLAKE
#endif // SIMSIMD_TARGET_X86


    // Normalize the vectors
    inline static float compute_normalized_factor(const float *vector, int dim) {
#if SIMSIMD_TARGET_NEON
        return compute_normalized_factor_neon(vector, dim);
#elif SIMSIMD_TARGET_SKYLAKE
            return compute_normalized_factor_skylake(vector, dim);
#elif SIMSIMD_TARGET_HASWELL
            return compute_normalized_factor_haswell(vector, dim);
#else
            float norm = 0;
            for (int i = 0; i < dim; i++) {
                norm += vector[i] * vector[i];
            }
            return 1.0f / std::sqrt(norm);
#endif
    }

    inline static void normalize_vector(const float *vector, int dim, float *norm_vector) {
#if SIMSIMD_TARGET_NEON
        normalize_vectors_neon(vector, dim, norm_vector);
#elif SIMSIMD_TARGET_SKYLAKE
            normalize_vectors_skylake(vector, dim, norm_vector);
#elif SIMSIMD_TARGET_HASWELL
            normalize_vectors_haswell(vector, dim, norm_vector);
#else
            float norm = compute_normalized_factor(vector, dim);
            for (int i = 0; i < dim; i++) {
                norm_vector[i] = vector[i] * norm;
            }
#endif
    }

    inline static void normalize_vectors(const float *vector, int dim, size_t n, float *norm_vector) {
        for (size_t i = 0; i < n; i++) {
            normalize_vector(vector + i * dim, dim, norm_vector + i * dim);
        }
    }
} // namespace orange
