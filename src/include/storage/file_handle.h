#pragma once

#include <string>

namespace orangedb::storage {
    // Not used as of now!!
    struct ReadRequest {
        std::unique_ptr<uint8_t *> buffer;
        uint64_t numBytes;
        uint64_t offset;
    };

    // TODO:
    // 1. Test Mmap
    // 2. Test PRead
    // 3. Test IO-uring
    // 4. Implement PM CSR
    class FileHandle {
    public:
        explicit FileHandle(std::string &path, int flag, bool readOnly);

        uint8_t *readOnlyMmap(uint64_t numBytes, uint64_t offset);

        void write(const uint8_t *buffer, uint64_t numBytes, uint64_t offset);

        ~FileHandle();

    private:
        int fd;
        std::string path;
    };
} // namespace orangedb::storage
