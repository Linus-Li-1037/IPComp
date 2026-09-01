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

// `stream` is the block's single component, already read.  The per-frame path read the
// whole file up front for the same reason: IPComp's layer index lives inside the stream,
// so the codec fetches layers within bytes it already holds.
template <class T>
bool reconstruct_frame(std::vector<uint8_t>& stream,
                       const IPCompFUN3D::FrameInfo& info,
                       const std::vector<double>& relative_tolerances,
                       MPI_Comm comm,
                       std::vector<unsigned long long>& retrieved_sizes,
                       std::vector<double>& reconstruct_times) {
    using Codec = SZ3::SZProgressiveMQuant<
        T, 1, SZ3::LinearQuantizer2<T>, SZ3::BypassEncoder<int>,
        SZ3::Lossless_zstd>;
    const std::array<size_t, 1> dimensions{
        static_cast<size_t>(info.num_elements)};
    Codec codec(SZ3::LinearQuantizer2<T>(info.num_elements, 1),
                SZ3::BypassEncoder<int>(), SZ3::Lossless_zstd(), dimensions,
                1, 0, 50000, info.layers, 0);

    if (stream.empty()) return false;

    // Restore the encoder's layer ladder directly from its local range.  This avoids
    // allocating an original-sized dummy field on every MPI rank.
    codec.setupLayersFromRange(static_cast<T>(info.value_range));
    for (size_t i = 0; i < relative_tolerances.size(); ++i) {
        std::vector<double> target{
            relative_tolerances[i] * info.value_range};
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        codec.progressive_reconstruct(
            reinterpret_cast<SZ3::uchar*>(stream.data()), nullptr, target);
        reconstruct_times[i] += MPI_Wtime() - start;
        retrieved_sizes[i] += codec.get_retrieved_size() +
                              sizeof(IPCompFUN3D::FrameInfo);
    }
    codec.release_workspace();
    return true;
}

template <class T>
int run(int argc, char** argv, MPI_Comm comm) {
    int rank = 0;
    int np = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);
    if (argc < 7) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s input_dir variables nt num_tolerances "
                            "tolerance... f|d\n", argv[0]);
        }
        return 1;
    }
    if (np < 72 || np % 72 != 0) return 1;

    const int num_timesteps = std::atoi(argv[3]);
    const int num_tolerances = std::atoi(argv[4]);
    if (num_timesteps < 1 || num_tolerances < 1 ||
        argc != 6 + num_tolerances) return 1;

    std::vector<double> tolerances(num_tolerances);
    for (int i = 0; i < num_tolerances; ++i) {
        tolerances[i] = std::atof(argv[5 + i]);
    }

    std::string input_root = argv[1];
    if (input_root.back() != '/') input_root += '/';
    const int subdomain = rank % 72;
    const auto variables = IPCompFUN3D::variables(
        IPCompFUN3D::subdomain_dir(input_root, subdomain), argv[2]);
    if (variables.empty()) return 1;

    std::vector<unsigned long long> local_retrieved(num_tolerances, 0);
    std::vector<unsigned long long> total_retrieved(num_tolerances, 0);
    std::vector<double> local_reconstruct_time(num_tolerances, 0);
    std::vector<double> max_reconstruct_time(num_tolerances, 0);
    unsigned long long local_num_elements = 0;

    // One file per rank: the block for (timestep, field) is found through the directory
    // at the front of it, so the read path needs no MPI at all.
    IPCompFUN3D::SingleFileArchive archive(
        IPCompFUN3D::single_file_archive_name(input_root, np, rank));
    if (!archive.open()) return 1;
    if (archive.num_fields() != variables.size() ||
        archive.num_timesteps() < static_cast<uint64_t>(num_timesteps)) {
        fprintf(stderr,
                "ERROR: rank %d asked for %d timesteps of %zu fields, %s holds "
                "%llu timesteps of %llu fields\n",
                rank, num_timesteps, variables.size(), archive.path().c_str(),
                static_cast<unsigned long long>(archive.num_timesteps()),
                static_cast<unsigned long long>(archive.num_fields()));
        return 1;
    }
    if (archive.header().element_size != sizeof(T)) {
        fprintf(stderr,
                "ERROR: %s was compressed from %u-byte values, this run uses %zu "
                "(f / d mismatch)\n",
                archive.path().c_str(), archive.header().element_size, sizeof(T));
        return 1;
    }

    for (int timestep = 0; timestep < num_timesteps; ++timestep) {
        for (size_t field = 0; field < variables.size(); ++field) {
            std::vector<uint8_t> metadata;
            std::vector<uint8_t> stream;
            if (!archive.read_block(
                    IPCompFUN3D::frame_block_index(timestep, field, variables.size()),
                    metadata, stream)) {
                return 1;
            }
            if (metadata.size() < sizeof(IPCompFUN3D::FrameInfo)) {
                fprintf(stderr, "ERROR: rank %d read a block of %s with no FrameInfo\n",
                        rank, archive.path().c_str());
                return 1;
            }
            IPCompFUN3D::FrameInfo info;
            memcpy(&info, metadata.data(), sizeof(info));

            if (!reconstruct_frame<T>(
                    stream, info, tolerances, comm, local_retrieved,
                    local_reconstruct_time)) return 1;
            local_num_elements += info.num_elements;
        }
    }

    unsigned long long total_num_elements = 0;
    MPI_Reduce(&local_num_elements, &total_num_elements, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(local_retrieved.data(), total_retrieved.data(), num_tolerances,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(local_reconstruct_time.data(), max_reconstruct_time.data(),
               num_tolerances, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        printf("IPComp ranks=%d preprocessing=0.000000\n", np);
        for (int i = 0; i < num_tolerances; ++i) {
            printf("  rel_tol=%.3g retrieved=%llu bitrate=%.4f ratio=%.4f "
                   "reconstruct=%.6f\n",
                   tolerances[i], total_retrieved[i],
                   total_retrieved[i] * 8.0 / total_num_elements,
                   static_cast<double>(total_num_elements) * sizeof(T) /
                       total_retrieved[i],
                   max_reconstruct_time[i]);
        }
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
