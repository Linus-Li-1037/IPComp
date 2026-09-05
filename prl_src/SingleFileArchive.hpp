#ifndef IPCOMP_SINGLE_FILE_ARCHIVE_HPP
#define IPCOMP_SINGLE_FILE_ARCHIVE_HPP

// One refactored file per MPI rank, instead of two files per frame.
//
// The per-frame layout costs a compressed stream plus an info file for every timestep,
// every field and every rank: 2 * nt * nf files per rank.  For FUN3D at 72 subdomains
// that is on the order of 100k files, and ~800k at 8 partitions per subdomain -- a load
// no parallel filesystem handles well, since each file costs metadata-server operations
// that dwarf the write itself.  This packs everything a rank owns into ONE file:
//
//   [ archive header ]  64 bytes, fixed
//   [ block directory ] num_blocks * {uint64 offset, uint64 size}, RESERVED up front so
//                       the writer never moves a block and never has to communicate
//   [ block 0 ][ block 1 ] ...
//
// One block is one (timestep, field) of one rank, at index
// single_file_block_index(timestep, field, num_fields).  The index is a pure function of
// (timestep, field) and each rank writes only its own file, so neither side needs any
// MPI: the reader seeks straight to directory[index].
//
// A block is self-contained -- FrameInfo and the stream, nothing beside it:
//
//   [ block header ]  uint64 metadata_size, uint64 num_components,
//                     then uint64 component_size[num_components]
//   [ metadata ]      FrameInfo: what the reader needs before it can turn a relative
//                     tolerance into an absolute one
//   [ component 0 ]   the IPComp stream, exactly the bytes the per-frame file held.
//                     IPComp's own layer index lives inside it, so the reader takes the
//                     whole component and lets the codec fetch layers within it --
//                     which is what the per-frame path did too.
//
// Offsets are uint64 throughout: a rank's file is every timestep and field it owns
// summed together and overruns uint32 easily.

#include <sys/types.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace IPCompFUN3D {

// "SFARCH01" in file order.  Guards against pointing a reader at a per-frame stream or
// at a stale format instead of computing nonsense offsets from its bytes.
constexpr uint64_t SINGLE_FILE_MAGIC = 0x3130484352414653ULL;
constexpr uint32_t SINGLE_FILE_VERSION = 1;

constexpr uint64_t SINGLE_FILE_HEADER_BYTES = 64;
constexpr uint64_t SINGLE_FILE_DIRECTORY_ENTRY_BYTES = 16;

inline void sfa_put(uint8_t*& pos, uint64_t value) {
    std::memcpy(pos, &value, sizeof(value));
    pos += sizeof(value);
}
inline void sfa_put32(uint8_t*& pos, uint32_t value) {
    std::memcpy(pos, &value, sizeof(value));
    pos += sizeof(value);
}
inline uint64_t sfa_get(const uint8_t*& pos) {
    uint64_t value = 0;
    std::memcpy(&value, pos, sizeof(value));
    pos += sizeof(value);
    return value;
}
inline uint32_t sfa_get32(const uint8_t*& pos) {
    uint32_t value = 0;
    std::memcpy(&value, pos, sizeof(value));
    pos += sizeof(value);
    return value;
}

// Blocks are laid out field-fastest, so one timestep of a rank is contiguous.
inline uint64_t single_file_block_index(uint64_t timestep, uint64_t field,
                                       uint64_t num_fields) {
    return timestep * num_fields + field;
}

// A rank's whole refactored output.  np and rank are in the name so one directory can
// hold several rank counts, exactly as the per-frame names did.
inline std::string single_file_archive_name(const std::string& directory, int np,
                                           int rank) {
    std::string result = directory;
    if (!result.empty() && result.back() != '/') result += '/';
    return result + "ipcomp.p" + std::to_string(np) + ".rank" +
           std::to_string(rank) + ".bin";
}

struct SingleFileArchiveHeader {
    uint64_t magic = SINGLE_FILE_MAGIC;
    uint32_t version = SINGLE_FILE_VERSION;
    uint32_t element_size = 0;      // sizeof(T) the fields were compressed from
    uint64_t num_timesteps = 0;
    uint64_t num_fields = 0;
    uint64_t num_blocks = 0;        // num_timesteps * num_fields
    uint64_t directory_offset = SINGLE_FILE_HEADER_BYTES;

    // Where the first block starts: past the header and the reserved directory.
    uint64_t payload_offset() const {
        return directory_offset + num_blocks * SINGLE_FILE_DIRECTORY_ENTRY_BYTES;
    }

    void serialize(uint8_t* buffer) const {
        std::memset(buffer, 0, SINGLE_FILE_HEADER_BYTES);
        uint8_t* pos = buffer;
        sfa_put(pos, magic);
        sfa_put32(pos, version);
        sfa_put32(pos, element_size);
        sfa_put(pos, num_timesteps);
        sfa_put(pos, num_fields);
        sfa_put(pos, num_blocks);
        sfa_put(pos, directory_offset);
    }

    bool deserialize(const uint8_t* buffer) {
        const uint8_t* pos = buffer;
        magic = sfa_get(pos);
        version = sfa_get32(pos);
        element_size = sfa_get32(pos);
        num_timesteps = sfa_get(pos);
        num_fields = sfa_get(pos);
        num_blocks = sfa_get(pos);
        directory_offset = sfa_get(pos);
        return magic == SINGLE_FILE_MAGIC && version == SINGLE_FILE_VERSION &&
               num_blocks == num_timesteps * num_fields;
    }
};

// Everything needed to seek inside one block.  Sizes come from the block header; the
// offsets are absolute in the file, so reading the stream is one seek.
struct SingleFileBlockLayout {
    uint64_t block_offset = 0;
    uint64_t block_size = 0;
    uint64_t header_size = 0;
    uint64_t metadata_offset = 0;
    uint64_t metadata_size = 0;
    std::vector<uint64_t> component_sizes;
    std::vector<uint64_t> component_offsets;

    static uint64_t header_bytes(uint64_t num_components) {
        return 2 * sizeof(uint64_t) + num_components * sizeof(uint64_t);
    }

    void finish() {
        header_size = header_bytes(component_sizes.size());
        metadata_offset = block_offset + header_size;
        component_offsets.resize(component_sizes.size());
        uint64_t pos = metadata_offset + metadata_size;
        for (size_t i = 0; i < component_sizes.size(); ++i) {
            component_offsets[i] = pos;
            pos += component_sizes[i];
        }
        block_size = pos - block_offset;
    }
};

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------
// A block is buffered and flushed in ONE sequential fwrite.  The directory is reserved
// at open() and patched at close(), so nothing seeks backwards in between.
//
// An empty path makes the writer a dry run: the block is still assembled and its size
// still comes back exact, but no file is touched -- what write_mode == 0 uses to measure
// the compression without paying for output.
class SingleFileWriter {
public:
    SingleFileWriter(const std::string& path, uint64_t num_timesteps,
                     uint64_t num_fields, uint32_t element_size)
        : path_(path) {
        header_.element_size = element_size;
        header_.num_timesteps = num_timesteps;
        header_.num_fields = num_fields;
        header_.num_blocks = num_timesteps * num_fields;
    }

    ~SingleFileWriter() {
        if (file_) {
            std::fclose(file_);
            file_ = nullptr;
        }
    }

    SingleFileWriter(const SingleFileWriter&) = delete;
    SingleFileWriter& operator=(const SingleFileWriter&) = delete;

    bool dry_run() const { return path_.empty(); }

    bool open() {
        directory_.assign(2 * header_.num_blocks, 0);
        cursor_ = header_.payload_offset();
        if (dry_run()) return true;

        file_ = std::fopen(path_.c_str(), "wb");
        if (!file_) {
            std::cerr << "ERROR: cannot create " << path_ << std::endl;
            return false;
        }
        std::vector<uint8_t> prologue(header_.payload_offset(), 0);
        header_.serialize(prologue.data());
        return write_bytes(prologue.data(), prologue.size());
    }

    // Start a fresh block.  Must be called before the frame's component: a leftover
    // component from the previous frame would be written again.
    void begin_block() { components_.clear(); }

    // Copies the bytes, so the caller may free its workspace immediately after.
    void add_component(const void* data, size_t size) {
        const auto* begin = static_cast<const uint8_t*>(data);
        components_.emplace_back(begin, begin + size);
    }

    // Flushes the buffered component and the metadata as block `index`.  `block_size`
    // comes back with the bytes the block occupies, its header included.
    bool commit_block(uint64_t index, const void* metadata, size_t metadata_size,
                      size_t& block_size) {
        if (index >= header_.num_blocks) {
            std::cerr << "ERROR: block " << index << " is outside the "
                      << header_.num_blocks << " reserved for " << path_ << std::endl;
            return false;
        }

        SingleFileBlockLayout layout;
        layout.block_offset = cursor_;
        layout.metadata_size = metadata_size;
        layout.component_sizes.resize(components_.size());
        for (size_t i = 0; i < components_.size(); ++i) {
            layout.component_sizes[i] = components_[i].size();
        }
        layout.finish();

        std::vector<uint8_t> block(layout.block_size);
        uint8_t* pos = block.data();
        sfa_put(pos, layout.metadata_size);
        sfa_put(pos, static_cast<uint64_t>(layout.component_sizes.size()));
        for (uint64_t size : layout.component_sizes) sfa_put(pos, size);
        if (metadata_size) {
            std::memcpy(pos, metadata, metadata_size);
            pos += metadata_size;
        }
        for (const auto& component : components_) {
            if (component.empty()) continue;
            std::memcpy(pos, component.data(), component.size());
            pos += component.size();
        }

        if (!dry_run() && !write_bytes(block.data(), block.size())) return false;

        directory_[2 * index] = layout.block_offset;
        directory_[2 * index + 1] = layout.block_size;
        cursor_ += layout.block_size;
        block_size = layout.block_size;
        components_.clear();
        return true;
    }

    // Patches the reserved directory.  Must be called, or the archive has a directory of
    // zeros and no block can be found.
    bool close() {
        if (dry_run()) return true;
        if (!file_) return false;

        std::vector<uint8_t> table(header_.num_blocks *
                                   SINGLE_FILE_DIRECTORY_ENTRY_BYTES);
        uint8_t* pos = table.data();
        for (uint64_t value : directory_) sfa_put(pos, value);

        bool ok = true;
        const auto io_start = std::chrono::steady_clock::now();
        if (fseeko(file_, static_cast<off_t>(header_.directory_offset), SEEK_SET) != 0) {
            std::cerr << "ERROR: cannot seek to the block directory of " << path_
                      << std::endl;
            ok = false;
        } else if (!table.empty() &&
                   std::fwrite(table.data(), 1, table.size(), file_) != table.size()) {
            std::cerr << "ERROR: short write of the block directory of " << path_
                      << std::endl;
            ok = false;
        }
        io_time_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - io_start).count();

        if (std::fclose(file_) != 0) {
            std::cerr << "ERROR: cannot close " << path_ << std::endl;
            ok = false;
        }
        file_ = nullptr;
        return ok;
    }

    uint64_t file_size() const { return cursor_; }
    uint64_t prologue_size() const { return header_.payload_offset(); }
    double io_time() const { return io_time_; }

private:
    bool write_bytes(const uint8_t* data, size_t size) {
        const auto io_start = std::chrono::steady_clock::now();
        const size_t written = std::fwrite(data, 1, size, file_);
        io_time_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - io_start).count();
        if (written != size) {
            std::cerr << "ERROR: short write of " << size << " bytes to " << path_
                      << std::endl;
            return false;
        }
        return true;
    }

    std::string path_;
    SingleFileArchiveHeader header_;
    FILE* file_ = nullptr;
    std::vector<uint64_t> directory_;                  // {offset, size} per block
    std::vector<std::vector<uint8_t>> components_;     // current frame
    uint64_t cursor_ = 0;
    double io_time_ = 0;
};

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------
// Owns the FILE* and the directory.  Opened once per rank, so nt * nf frames cost ONE
// fopen rather than two per frame.
class SingleFileArchive {
public:
    explicit SingleFileArchive(const std::string& path) : path_(path) {}
    ~SingleFileArchive() { close(); }

    SingleFileArchive(const SingleFileArchive&) = delete;
    SingleFileArchive& operator=(const SingleFileArchive&) = delete;

    bool open() {
        file_ = std::fopen(path_.c_str(), "rb");
        if (!file_) {
            std::cerr << "ERROR: cannot open " << path_ << std::endl;
            return false;
        }
        uint8_t prologue[SINGLE_FILE_HEADER_BYTES];
        if (!read_at(0, prologue, sizeof(prologue))) return false;
        if (!header_.deserialize(prologue)) {
            std::cerr << "ERROR: " << path_ << " is not an IPComp single-file archive"
                      << std::endl;
            return false;
        }
        std::vector<uint8_t> table(header_.num_blocks *
                                   SINGLE_FILE_DIRECTORY_ENTRY_BYTES);
        if (!table.empty() &&
            !read_at(header_.directory_offset, table.data(), table.size())) {
            return false;
        }
        directory_.resize(2 * header_.num_blocks);
        const uint8_t* pos = table.data();
        for (auto& value : directory_) value = sfa_get(pos);
        return true;
    }

    void close() {
        if (file_) {
            std::fclose(file_);
            file_ = nullptr;
        }
    }

    const SingleFileArchiveHeader& header() const { return header_; }
    uint64_t num_blocks() const { return header_.num_blocks; }
    uint64_t num_timesteps() const { return header_.num_timesteps; }
    uint64_t num_fields() const { return header_.num_fields; }
    const std::string& path() const { return path_; }
    double io_time() const { return io_time_; }

    bool block_layout(uint64_t index, SingleFileBlockLayout& layout) const {
        if (index >= header_.num_blocks) {
            std::cerr << "ERROR: block " << index << " is outside the "
                      << header_.num_blocks << " in " << path_ << std::endl;
            return false;
        }
        const uint64_t offset = directory_[2 * index];
        const uint64_t size = directory_[2 * index + 1];
        if (size == 0) {
            std::cerr << "ERROR: block " << index << " of " << path_
                      << " is empty; the archive was not closed cleanly" << std::endl;
            return false;
        }

        uint64_t fixed[2];
        if (!read_at(offset, fixed, sizeof(fixed))) return false;
        const uint64_t num_components = fixed[1];

        std::vector<uint64_t> component_sizes(num_components);
        if (num_components &&
            !read_at(offset + sizeof(fixed), component_sizes.data(),
                     num_components * sizeof(uint64_t))) {
            return false;
        }

        layout = SingleFileBlockLayout();
        layout.block_offset = offset;
        layout.metadata_size = fixed[0];
        layout.component_sizes = component_sizes;
        layout.finish();

        if (layout.block_size != size) {
            std::cerr << "ERROR: block " << index << " of " << path_ << " says "
                      << layout.block_size << " bytes, the directory says " << size
                      << std::endl;
            return false;
        }
        return true;
    }

    bool read_at(uint64_t offset, void* destination, size_t size) const {
        if (!file_) {
            std::cerr << "ERROR: " << path_ << " is not open" << std::endl;
            return false;
        }
        if (size == 0) return true;

        const auto io_start = std::chrono::steady_clock::now();
        bool ok = true;
        if (fseeko(file_, static_cast<off_t>(offset), SEEK_SET) != 0) {
            std::cerr << "ERROR: cannot seek to " << offset << " in " << path_
                      << std::endl;
            ok = false;
        } else if (std::fread(destination, 1, size, file_) != size) {
            std::cerr << "ERROR: short read of " << size << " bytes at " << offset
                      << " in " << path_ << std::endl;
            ok = false;
        }
        io_time_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - io_start).count();
        return ok;
    }

    // Reads one block: its metadata and its single component.  Both are needed in full
    // before the codec can do anything, so there is no partial-read path here.
    bool read_block(uint64_t index, std::vector<uint8_t>& metadata,
                    std::vector<uint8_t>& component) const {
        std::vector<std::vector<uint8_t>> components;
        if (!read_block_components(index, metadata, components)) return false;
        if (components.empty()) {
            std::cerr << "ERROR: block " << index << " of " << path_
                      << " has no stream" << std::endl;
            return false;
        }
        component = std::move(components[0]);
        return true;
    }

    // Reads one block whole: its metadata and every component.  A bundle block carries
    // two (the range table and the touched bytes), an archive block one.
    bool read_block_components(uint64_t index, std::vector<uint8_t>& metadata,
                               std::vector<std::vector<uint8_t>>& components) const {
        SingleFileBlockLayout layout;
        if (!block_layout(index, layout)) return false;
        metadata.assign(layout.metadata_size, 0);
        if (!metadata.empty() &&
            !read_at(layout.metadata_offset, metadata.data(), metadata.size())) {
            return false;
        }
        components.assign(layout.component_sizes.size(), std::vector<uint8_t>());
        for (size_t c = 0; c < components.size(); ++c) {
            components[c].assign(layout.component_sizes[c], 0);
            if (!components[c].empty() &&
                !read_at(layout.component_offsets[c], components[c].data(),
                         components[c].size())) {
                return false;
            }
        }
        return true;
    }

private:
    std::string path_;
    FILE* file_ = nullptr;
    SingleFileArchiveHeader header_;
    std::vector<uint64_t> directory_;
    mutable double io_time_ = 0;
};

}  // namespace IPCompFUN3D

#endif
