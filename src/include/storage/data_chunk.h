#pragma once

#include <vector>
#include "file_handle.h"
#include "common.h"

namespace orangedb::storage {
    struct DataChunkMetadata {
        uint64_t numBytesPerElement;
        uint64_t startOffset;
        uint64_t numElements;
    };

    class DataChunk {
    public:
        virtual void resize(uint64_t size) = 0;

        virtual uint8_t *getBufferUnsafe() = 0;

        virtual uint64_t size() const = 0;
    };

    // Sequential data chunk in memory
    class InMemoryDataChunk : public DataChunk {
    public:
        explicit InMemoryDataChunk(DataChunkMetadata *metadata) : metadata(metadata) {
            allocAligned(((void **) &buffer), metadata->numBytesPerElement * metadata->numElements, 512);
        }

        inline void resize(uint64_t newSize) override {
            std::lock_guard<std::mutex> lock(mtx);
            if (newSize <= metadata->numElements) {
                metadata->numElements = newSize;
                return;
            }

            uint8_t *newBuffer;
            allocAligned(((void **) &newBuffer), metadata->numBytesPerElement * newSize, 512);
            memcpy(newBuffer, buffer, metadata->numBytesPerElement * metadata->numElements);
            free(buffer);
            buffer = newBuffer;
            metadata->numElements = newSize;
        }

        inline uint8_t *getBufferUnsafe() override {
            return buffer;
        }

        inline uint64_t size() const {
            return metadata->numElements;
        }

        // Flush buffer to disk
        inline void flush(FileHandle &handle) {
            std::lock_guard<std::mutex> lock(mtx);
            handle.write(buffer, metadata->numBytesPerElement * metadata->numElements, metadata->startOffset);
        }

        ~InMemoryDataChunk() {
            free(buffer);
        }

    private:
        uint8_t *buffer;
        DataChunkMetadata *metadata;
        std::mutex mtx;
    };

    // Mmap read-only data chunk from disk
    class OnDiskDataChunk : public DataChunk {
    public:
        explicit OnDiskDataChunk(DataChunkMetadata *metadata, FileHandle &handle) : metadata(metadata) {
            mappedBuffer = handle.readOnlyMmap(metadata->numBytesPerElement * metadata->numElements,
                                               metadata->startOffset);
        }

        ~OnDiskDataChunk() = default;

        inline void resize(uint64_t newSize) {
            throw std::runtime_error("Cannot resize on-disk data chunk");
        }

        inline uint8_t *getBufferUnsafe() {
            return mappedBuffer;
        }

        inline uint64_t size() const {
            return metadata->numElements;
        }

    private:
        DataChunkMetadata *metadata;
        uint8_t *mappedBuffer;
    };
} // namespace orangedb::storage
