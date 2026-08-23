#ifndef IPCOMP_FUN3D_PARALLEL_HPP
#define IPCOMP_FUN3D_PARALLEL_HPP

#include <mpi.h>

#include <cerrno>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>

#include <SZ3/utils/Iterator.hpp>

namespace IPCompFUN3D {

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
