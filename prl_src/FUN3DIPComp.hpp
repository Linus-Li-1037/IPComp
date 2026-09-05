#ifndef IPCOMP_FUN3D_PARALLEL_HPP
#define IPCOMP_FUN3D_PARALLEL_HPP

#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <utility>
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>

#include <SZ3/utils/Iterator.hpp>
#include <SZ3/utils/FileUtil.hpp>

#include "SingleFileArchive.hpp"

namespace IPCompFUN3D {

inline uint64_t frame_block_index(int timestep, size_t field, size_t num_fields) {
    return single_file_block_index(static_cast<uint64_t>(timestep),
                                  static_cast<uint64_t>(field),
                                  static_cast<uint64_t>(num_fields));
}

// ---------------------------------------------------------------------------
// Retrieve-then-transfer.  A retrieve stage on the source cluster writes a BUNDLE for
// one tolerance -- the same archive layout, one file per rank -- and a reconstruct stage
// on the other cluster reads it.
//
// IPComp selects its layers inside the codec (calcBitgroup / strategy over a size table
// at the front of each layer's stream, with the unpredictable block at the end), so the
// stream cannot be cut down by reasoning about it from outside.  Instead the retrieve
// stage RUNS one reconstruction with the codec's access log switched on and learns
// exactly which byte ranges it consumed.  The bundle block then carries two components:
//   component 0: the range table  -- uint64 full_size, uint64 n, then n x {offset, len}
//   component 1: the bytes of those ranges, concatenated in table order
// The reconstruct stage rebuilds a full-size buffer with the ranges put back at their
// offsets (the rest zero) and hands it to the stock codec, whose selection -- driven by
// the header blocks, which are always among the ranges -- lands on the same blocks.  It
// logs again and refuses to trust a result if the codec read anything outside the ranges
// shipped, rather than decoding zeros silently.
// ---------------------------------------------------------------------------

using ByteRange = std::pair<size_t, size_t>;   // offset, length

// Sorts and coalesces overlapping or adjacent ranges.  The codec logs each layer's
// header block once per loadcfg call, so duplicates are the norm.
inline std::vector<ByteRange> merge_ranges(std::vector<ByteRange> ranges) {
    std::sort(ranges.begin(), ranges.end());
    std::vector<ByteRange> merged;
    for (const auto& r : ranges) {
        if (r.second == 0) continue;
        if (!merged.empty() && r.first <= merged.back().first + merged.back().second) {
            const size_t end = std::max(merged.back().first + merged.back().second,
                                        r.first + r.second);
            merged.back().second = end - merged.back().first;
        } else {
            merged.push_back(r);
        }
    }
    return merged;
}

inline bool ranges_cover(const std::vector<ByteRange>& shipped, const ByteRange& r) {
    for (const auto& s : shipped) {
        if (s.first <= r.first && r.first + r.second <= s.first + s.second) return true;
    }
    return r.second == 0;
}

inline std::vector<uint8_t> serialize_range_table(size_t full_size,
                                                  const std::vector<ByteRange>& ranges) {
    std::vector<uint8_t> out((2 + 2 * ranges.size()) * sizeof(uint64_t));
    uint8_t* pos = out.data();
    sfa_put(pos, full_size);
    sfa_put(pos, ranges.size());
    for (const auto& r : ranges) {
        sfa_put(pos, r.first);
        sfa_put(pos, r.second);
    }
    return out;
}

inline bool deserialize_range_table(const std::vector<uint8_t>& table, size_t& full_size,
                                    std::vector<ByteRange>& ranges) {
    if (table.size() < 2 * sizeof(uint64_t)) return false;
    const uint8_t* pos = table.data();
    full_size = sfa_get(pos);
    const uint64_t n = sfa_get(pos);
    if (table.size() != (2 + 2 * n) * sizeof(uint64_t)) return false;
    ranges.resize(n);
    for (auto& r : ranges) {
        r.first = sfa_get(pos);
        r.second = sfa_get(pos);
    }
    return true;
}

inline std::string retrieved_archive_name(const std::string& directory, int np, int rank) {
    std::string result = directory;
    if (!result.empty() && result.back() != '/') result += '/';
    return result + "ipcomp.retrieved.p" + std::to_string(np) + ".rank" +
           std::to_string(rank) + ".bin";
}

inline void put_f64(uint8_t*& pos, double value) {
    std::memcpy(pos, &value, sizeof(value));
    pos += sizeof(value);
}
inline double get_f64(const uint8_t*& pos) {
    double value = 0;
    std::memcpy(&value, pos, sizeof(value));
    pos += sizeof(value);
    return value;
}

// What the retrieve stage decided for one frame.  It rides at the END of the block's
// metadata in the bundle, after FrameInfo, followed by its own byte count, so the
// reconstruct stage peels it off the tail.  taken = { stream bytes shipped }.
struct RetrievalRecord {
    static constexpr uint64_t MAGIC = 0x3143455256525452ULL;   // "RTRVREC1" in file order

    double rel_tolerance = 0;
    double abs_tolerance = 0;
    std::vector<uint64_t> taken;

    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> out(4 * sizeof(uint64_t) + taken.size() * sizeof(uint64_t));
        uint8_t* pos = out.data();
        sfa_put(pos, MAGIC);
        put_f64(pos, rel_tolerance);
        put_f64(pos, abs_tolerance);
        sfa_put(pos, static_cast<uint64_t>(taken.size()));
        for (uint64_t t : taken) sfa_put(pos, t);
        return out;
    }

    bool deserialize(const uint8_t* data, size_t size) {
        if (size < 4 * sizeof(uint64_t)) return false;
        const uint8_t* pos = data;
        if (sfa_get(pos) != MAGIC) return false;
        rel_tolerance = get_f64(pos);
        abs_tolerance = get_f64(pos);
        const uint64_t count = sfa_get(pos);
        if (size != 4 * sizeof(uint64_t) + count * sizeof(uint64_t)) return false;
        taken.resize(count);
        for (auto& t : taken) t = sfa_get(pos);
        return true;
    }
};

// bundle metadata = FrameInfo ++ record ++ uint64(record bytes)
inline std::vector<uint8_t> append_record(const std::vector<uint8_t>& metadata,
                                          const RetrievalRecord& record) {
    const auto bytes = record.serialize();
    std::vector<uint8_t> out(metadata);
    out.insert(out.end(), bytes.begin(), bytes.end());
    uint8_t tail[sizeof(uint64_t)];
    uint8_t* pos = tail;
    sfa_put(pos, static_cast<uint64_t>(bytes.size()));
    out.insert(out.end(), tail, tail + sizeof(tail));
    return out;
}

// Inverse: peels the record off the tail.
inline bool split_record(const std::vector<uint8_t>& bundle_metadata,
                         RetrievalRecord& record) {
    if (bundle_metadata.size() < sizeof(uint64_t)) return false;
    const uint8_t* pos = bundle_metadata.data() + bundle_metadata.size() - sizeof(uint64_t);
    const uint64_t record_size = sfa_get(pos);
    if (record_size + sizeof(uint64_t) > bundle_metadata.size()) return false;
    return record.deserialize(
        bundle_metadata.data() + bundle_metadata.size() - sizeof(uint64_t) - record_size,
        record_size);
}

struct FrameInfo {
    uint64_t num_elements = 0;
    double value_range = 0;
    double local_value_range = 0;
    int32_t layers = 0;
};

inline bool ensure_directory(const std::string& path) {
    std::string current;
    for (char c : path) {
        current.push_back(c);
        if (c == '/' && current != "/" &&
            ::mkdir(current.c_str(), 0775) != 0 && errno != EEXIST) {
            return false;
        }
    }
    return current.empty() || current.back() == '/' ||
           ::mkdir(current.c_str(), 0775) == 0 || errno == EEXIST;
}

inline std::string subdomain_dir(const std::string& root, int subdomain) {
    std::string directory = root;
    if (directory.back() != '/') {
        directory += '/';
    }
    return directory + "subdomain" + std::to_string(subdomain + 1) + '/';
}

inline std::vector<std::string> variables(const std::string& directory,
                                          const std::string& specification) {
    if (specification != "all") {
        std::vector<std::string> result;
        size_t begin = 0;
        while (begin <= specification.size()) {
            const size_t end = specification.find(',', begin);
            result.push_back(specification.substr(begin, end - begin));
            if (end == std::string::npos) {
                break;
            }
            begin = end + 1;
        }
        return result;
    }

    std::ifstream input(directory + "metadata.json");
    std::string text((std::istreambuf_iterator<char>(input)),
                     std::istreambuf_iterator<char>());
    const size_t key = text.find("\"variables\"");
    const size_t left = text.find('[', key);
    const size_t right = text.find(']', left);
    std::vector<std::string> result;
    if (key == std::string::npos || left == std::string::npos ||
        right == std::string::npos) {
        return result;
    }

    for (size_t position = left; position < right;) {
        const size_t first = text.find('"', position + 1);
        if (first == std::string::npos || first >= right) {
            break;
        }
        const size_t second = text.find('"', first + 1);
        result.push_back(text.substr(first + 1, second - first - 1));
        position = second;
    }
    return result;
}

inline std::string frame_base(const std::string& output_dir,
                              const std::string& variable,
                              int timestep,
                              int np,
                              int rank) {
    return output_dir + variable + ".dat." + std::to_string(timestep) +
           ".ipcomp.p" + std::to_string(np) + ".rank" +
           std::to_string(rank);
}

template <class T>
bool read_local_field(const std::string& data_file,
                      const std::string& partition_prefix,
                      int local_partition,
                      int partitions_per_subdomain,
                      std::vector<T>& local) {
    size_t num_values = 0;
    auto full = SZ3::readfile<T>(data_file.c_str(), num_values);
    if (!full || num_values == 0) {
        return false;
    }

    if (partitions_per_subdomain == 1) {
        local.assign(full.get(), full.get() + num_values);
        return true;
    }

    size_t num_partition_entries = 0;
    auto partition = SZ3::readfile<int32_t>(
        (partition_prefix + ".part").c_str(), num_partition_entries);
    if (!partition || num_partition_entries != num_values) {
        return false;
    }

    for (size_t i = 0; i < num_values; ++i) {
        if (partition[i] == local_partition) {
            local.push_back(full[i]);
        }
    }
    return !local.empty();
}

template <class T>
double global_range(const std::vector<T>& local, MPI_Comm comm) {
    T local_min = local.front();
    T local_max = local.front();
    for (T value : local) {
        if (value < local_min) local_min = value;
        if (value > local_max) local_max = value;
    }

    double min_value = local_min;
    double max_value = local_max;
    double global_min = 0;
    double global_max = 0;
    MPI_Allreduce(&min_value, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&max_value, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    return global_max - global_min;
}

template <class T>
double local_range(const std::vector<T>& local) {
    T minimum = local.front();
    T maximum = local.front();
    for (T value : local) {
        if (value < minimum) minimum = value;
        if (value > maximum) maximum = value;
    }
    return static_cast<double>(maximum) - minimum;
}

inline bool copy_metadata(const std::string& source_dir,
                          const std::string& output_root,
                          int subdomain,
                          int local_partition) {
    if (local_partition != 0) {
        return true;
    }
    const std::string destination =
        subdomain_dir(output_root, subdomain);
    if (!ensure_directory(destination)) {
        return false;
    }
    std::ifstream input(source_dir + "metadata.json", std::ios::binary);
    std::ofstream output(destination + "metadata.json",
                         std::ios::binary | std::ios::trunc);
    output << input.rdbuf();
    return input.good() || input.eof();
}

}  // namespace IPCompFUN3D

#endif
