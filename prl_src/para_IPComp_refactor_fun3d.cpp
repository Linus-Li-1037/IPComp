#include <mpi.h>

#include <SZ3/compressor/IPComp.hpp>
#include <SZ3/encoder/BypassEncoder.hpp>
#include <SZ3/lossless/Lossless_zstd.hpp>
#include <SZ3/quantizer/IntegerQuantizer2.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "FUN3DIPComp.hpp"

// Compresses one frame and buffers the stream as the block's single component.  The
// bytes are copied into the archive before the workspace is freed, so the codec's
// scratch allocation is not held any longer than it is now.
template <class T>
bool refactor_frame(const std::vector<T>& data, int layers, double value_range,
                    IPCompFUN3D::SingleFileWriter& writer,
                    size_t& compressed_size, double& compression_time) {
    using Codec = SZ3::SZProgressiveMQuant<
        T, 1, SZ3::LinearQuantizer2<T>, SZ3::BypassEncoder<int>,
        SZ3::Lossless_zstd>;
    const std::array<size_t, 1> dimensions{data.size()};
    Codec codec(SZ3::LinearQuantizer2<T>(data.size(), 1),
                SZ3::BypassEncoder<int>(), SZ3::Lossless_zstd(3), dimensions,
                1, 0, 50000, layers, 0);

    codec.setupLayersFromRange(static_cast<T>(value_range));
    const size_t capacity = static_cast<size_t>(
        (data.size() < 1000000 ? 100.0 : 2.0) * data.size()) * sizeof(T);
    auto* workspace = new SZ3::uchar[capacity];

    const double start = MPI_Wtime();
    SZ3::uchar* compressed = codec.compress(
        const_cast<T*>(data.data()), compressed_size, workspace);
    compression_time = MPI_Wtime() - start;
    writer.begin_block();
    writer.add_component(compressed, compressed_size);
    codec.release_workspace();
    delete[] workspace;
    return true;
}

template <class T>
int run(int argc, char** argv, MPI_Comm comm) {
    int rank = 0;
    int np = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);
    if (argc != 9) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s data_root output_dir variables nt "
                            "partition_prefix layers write_mode f|d\n", argv[0]);
        }
        return 1;
    }
    if (np < 72 || np % 72 != 0) {
        if (rank == 0) fprintf(stderr, "ERROR: ranks must be a multiple of 72\n");
        return 1;
    }

    std::string data_root = argv[1];
    std::string output_root = argv[2];
    if (data_root.back() != '/') data_root += '/';
    if (output_root.back() != '/') output_root += '/';
    const int num_timesteps = std::atoi(argv[4]);
    const int layers = std::atoi(argv[6]);
    const int write_mode = std::atoi(argv[7]);
    if (num_timesteps < 1 || layers < 1 ||
        (write_mode != 0 && write_mode != 1)) return 1;

    const int subdomain = rank % 72;
    const int local_partition = rank / 72;
    const int partitions_per_subdomain = np / 72;
    const std::string data_dir = IPCompFUN3D::subdomain_dir(data_root, subdomain);
    const auto variables = IPCompFUN3D::variables(data_dir, argv[3]);
    if (variables.empty()) return 1;

    std::string partition_prefix = argv[5];
    if (!partition_prefix.empty() && partition_prefix.front() != '/') {
        partition_prefix = data_dir + partition_prefix;
    }
    if (write_mode == 1 &&
        (!IPCompFUN3D::ensure_directory(output_root) ||
         !IPCompFUN3D::copy_metadata(data_dir, output_root, subdomain,
                                     local_partition))) return 1;

    // One file per rank holding every (timestep, field) it owns, instead of a stream and
    // an info file per frame.  write_mode 0 keeps it a dry run: the block is still
    // assembled and measured, but no file is created.
    IPCompFUN3D::SingleFileWriter writer(
        write_mode == 1
            ? IPCompFUN3D::single_file_archive_name(output_root, np, rank)
            : std::string(),
        static_cast<uint64_t>(num_timesteps),
        static_cast<uint64_t>(variables.size()), sizeof(T));
    if (!writer.open()) return 1;

    unsigned long long local_compressed_size = writer.prologue_size();
    unsigned long long local_num_elements = 0;
    double local_compression_time = 0;
    for (int timestep = 0; timestep < num_timesteps; ++timestep) {
        for (size_t field = 0; field < variables.size(); ++field) {
            const std::string& variable = variables[field];
            std::vector<T> local;
            const std::string input = data_dir + variable + ".dat." +
                                      std::to_string(timestep);
            if (!IPCompFUN3D::read_local_field(input, partition_prefix,
                    local_partition, partitions_per_subdomain, local)) {
                fprintf(stderr, "ERROR: rank %d cannot read %s\n", rank, input.c_str());
                return 1;
            }
            const double range = IPCompFUN3D::global_range(local, comm);

            MPI_Barrier(comm);
            size_t frame_size = 0;
            double frame_time = 0;
            if (!refactor_frame(local, layers, range, writer, frame_size,
                                frame_time)) return 1;

            IPCompFUN3D::FrameInfo info;
            info.num_elements = local.size();
            info.value_range = range;
            // Kept in the on-disk struct for compatibility; all new streams use
            // the same global range for both fields.
            info.local_value_range = range;
            info.layers = layers;

            // FrameInfo is the block's metadata, so a frame is one self-contained
            // object and the reader needs no sidecar file.
            size_t block_size = 0;
            if (!writer.commit_block(
                    IPCompFUN3D::frame_block_index(timestep, field, variables.size()),
                    &info, sizeof(info), block_size)) {
                return 1;
            }
            local_compressed_size += block_size;
            local_num_elements += local.size();
            local_compression_time += frame_time;
        }
    }
    if (!writer.close()) return 1;

    unsigned long long total_compressed_size = 0;
    unsigned long long total_num_elements = 0;
    double max_compression_time = 0;
    double local_write_time = writer.io_time();
    double max_write_time = 0;
    MPI_Reduce(&local_compressed_size, &total_compressed_size, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_num_elements, &total_num_elements, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_compression_time, &max_compression_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_write_time, &max_write_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("IPComp ranks=%d timesteps=%d fields=%zu preprocessing=0.000000 "
               "compression=%.6f write=%.6f files=%d blocks_per_file=%d "
               "total_compressed_size=%llu aggregate_CR=%.4f\n",
               np, num_timesteps, variables.size(), max_compression_time,
               max_write_time, write_mode == 1 ? np : 0,
               num_timesteps * static_cast<int>(variables.size()),
               total_compressed_size,
               static_cast<double>(total_num_elements) * sizeof(T) /
                   total_compressed_size);
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const char dtype = argc > 1 ? argv[argc - 1][0] : 'f';
    int result = 1;
    if (dtype == 'd') {
        result = run<double>(argc, argv, MPI_COMM_WORLD);
    } else if (dtype == 'f') {
        result = run<float>(argc, argv, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return result;
}
