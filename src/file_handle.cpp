#include "include/storage/file_handle.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>

namespace orangedb::storage {
    FileHandle::FileHandle(std::string &path, int flag, bool readOnly) : path(path) {
        if (readOnly) {
            flag |= O_RDONLY;
        } else {
            flag |= O_RDWR;
        }

        fd = open(path.c_str(), flag);
        if (fd == -1) {
            throw std::runtime_error("Failed to open file: " + path);
        }

        // Get the file size
        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            close(fd);
            throw std::runtime_error("Error getting file size: " + path);
        }
    }

    FileHandle::~FileHandle() {
        close(fd);
    }

    uint8_t *FileHandle::readOnlyMmap(uint64_t numBytes, uint64_t offset) {
        auto mapped = mmap(nullptr, numBytes, PROT_READ, MAP_PRIVATE, fd, offset);
        if (mapped == MAP_FAILED) {
            perror("Error mmapping the file");
            close(fd);
            throw std::runtime_error("Failed to open file: " + path);
        }
        return reinterpret_cast<uint8_t *>(mapped);
    }

    void FileHandle::write(const uint8_t *buffer, uint64_t numBytes, uint64_t offset) {
        if (pwrite(fd, buffer, numBytes, offset) == -1) {
            throw std::runtime_error("Error writing to file: " + path);
        }
    }
} // namespace orangedb::storage
