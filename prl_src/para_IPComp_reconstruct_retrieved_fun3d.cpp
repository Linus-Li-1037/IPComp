// Reconstruct stage of a retrieve-then-transfer workflow for IPComp.
//
// Reads the bundle para_IPComp_retrieve_fun3d wrote -- one file per rank; per frame the
// FrameInfo, the tolerance, a table of the byte ranges the codec consumed, and those
// bytes -- puts the ranges back at their offsets in a full-size buffer (the rest zero),
// and hands that to the stock codec at the recorded tolerance.  The codec's own layer
// selection is driven by the header blocks, which are always among the ranges, so it
// lands on the same blocks.  It is not trusted blindly: the access log runs again and
// any read outside the shipped ranges is an error, never a silent decode of zeros.
//
// verify=1 reads the original fields back and reports, per timestep and field, the
// largest absolute deviation against the bound the tolerance asked for.  The original is
// gathered the way the refactor gathered it: whole subdomain file for one partition per
// subdomain, else filtered by the .part file under partition_prefix.

#include <mpi.h>

#include <SZ3/compressor/IPComp.hpp>
#include <SZ3/encoder/BypassEncoder.hpp>
#include <SZ3/lossless/Lossless_zstd.hpp>
#include <SZ3/quantizer/IntegerQuantizer2.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "FUN3DIPComp.hpp"

// Largest absolute deviation over this rank's nodes; -1 if the lengths disagree.
template <class T>
double max_abs_error(const std::vector<T>& original, const std::vector<T>& reconstructed) {
    if (original.size() != reconstructed.size()) return -1;
    double worst = 0;
    for (size_t i = 0; i < original.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(original[i]) -
                                          static_cast<double>(reconstructed[i])));
    }
    return worst;
}

// Rebuilds the sparse stream and decodes it.  Returns false if the codec read outside
// what was shipped.  `decoded` receives the field when non-null.
template <class T>
bool reconstruct_frame(const std::vector<uint8_t>& table,
                       const std::vector<uint8_t>& bytes,
                       const IPCompFUN3D::FrameInfo& info,
                       double rel_tolerance,
                       MPI_Comm comm,
                       unsigned long long& retrieved_size,
                       double& reconstruct_time,
                       std::vector<T>* decoded,
                       std::string& error) {
    size_t full_size = 0;
    std::vector<IPCompFUN3D::ByteRange> shipped;
    if (!IPCompFUN3D::deserialize_range_table(table, full_size, shipped)) {
        error = "malformed range table";
        return false;
    }
    std::vector<uint8_t> stream(full_size, 0);
    size_t pos = 0;
    for (const auto& r : shipped) {
        if (r.first + r.second > full_size || pos + r.second > bytes.size()) {
            error = "range table does not fit the bytes shipped";
            return false;
        }
        memcpy(stream.data() + r.first, bytes.data() + pos, r.second);
        pos += r.second;
    }
    if (pos != bytes.size()) {
        error = "bytes shipped do not match the range table";
        return false;
    }

    using Codec = SZ3::SZProgressiveMQuant<
        T, 1, SZ3::LinearQuantizer2<T>, SZ3::BypassEncoder<int>,
        SZ3::Lossless_zstd>;
    const std::array<size_t, 1> dimensions{static_cast<size_t>(info.num_elements)};
    Codec codec(SZ3::LinearQuantizer2<T>(info.num_elements, 1),
                SZ3::BypassEncoder<int>(), SZ3::Lossless_zstd(), dimensions,
                1, 0, 50000, info.layers, 0);
    if (stream.empty()) {
        error = "empty stream";
        return false;
    }
    codec.setupLayersFromRange(static_cast<T>(info.value_range));

    std::vector<IPCompFUN3D::ByteRange> log;
    codec.set_access_log(&log);
    std::vector<double> target{rel_tolerance * info.value_range};
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    const T* out = codec.progressive_reconstruct(
        reinterpret_cast<SZ3::uchar*>(stream.data()), nullptr, target);
    reconstruct_time += MPI_Wtime() - start;
    codec.set_access_log(nullptr);
    retrieved_size += codec.get_retrieved_size() + sizeof(IPCompFUN3D::FrameInfo);
    // the codec owns `out`; copy before the workspace goes away
    if (decoded) decoded->assign(out, out + info.num_elements);
    codec.release_workspace();

    for (const auto& r : IPCompFUN3D::merge_ranges(std::move(log))) {
        if (!IPCompFUN3D::ranges_cover(shipped, r)) {
            error = "codec read [" + std::to_string(r.first) + ", " +
                    std::to_string(r.first + r.second) +
                    ") which the bundle does not hold";
            return false;
        }
    }
    return true;
}

template <class T>
int run(int argc, char** argv, MPI_Comm comm) {
    int rank = 0;
    int np = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);

    const char* usage =
        "Usage: %s bundle_dir variables nt verify [original_root partition_prefix] f|d\n"
        "  The tolerance is the one the bundle was retrieved for.\n"
        "  verify=1 checks each reconstruction against the original fields under "
        "original_root (partition_prefix selects this rank's nodes when there is more "
        "than one partition per subdomain); verify=0 omits both.\n";
    if (argc != 6 && argc != 8) {
        if (rank == 0) fprintf(stderr, usage, argv[0]);
        return 1;
    }
    const int num_timesteps = std::atoi(argv[3]);
    const int verify = std::atoi(argv[4]);
    if (np < 72 || np % 72 != 0 || num_timesteps < 1 || (verify != 0 && verify != 1) ||
        argc != 6 + 2 * verify) {
        if (rank == 0) fprintf(stderr, usage, argv[0]);
        return 1;
    }

    std::string bundle_root = argv[1];
    if (bundle_root.back() != '/') bundle_root += '/';
    const int subdomain = rank % 72;
    const int local_partition = rank / 72;
    const int partitions_per_subdomain = np / 72;
    const auto variables = IPCompFUN3D::variables(
        IPCompFUN3D::subdomain_dir(bundle_root, subdomain), argv[2]);
    if (variables.empty()) return 1;

    std::string original_dir;
    std::string partition_prefix;
    if (verify == 1) {
        std::string original_root = argv[5];
        if (original_root.back() != '/') original_root += '/';
        original_dir = IPCompFUN3D::subdomain_dir(original_root, subdomain);
        partition_prefix = argv[6];
        if (!partition_prefix.empty() && partition_prefix.front() != '/') {
            partition_prefix = original_dir + partition_prefix;
        }
    }

    IPCompFUN3D::SingleFileArchive archive(
        IPCompFUN3D::retrieved_archive_name(bundle_root, np, rank));
    if (!archive.open()) return 1;
    if (archive.num_fields() != variables.size() ||
        archive.num_timesteps() < static_cast<uint64_t>(num_timesteps) ||
        archive.header().element_size != sizeof(T)) {
        fprintf(stderr, "ERROR: rank %d: %s does not match this run "
                        "(fields, timesteps or f/d)\n",
                rank, archive.path().c_str());
        return 1;
    }

    const size_t num_frames = static_cast<size_t>(num_timesteps) * variables.size();
    unsigned long long local_retrieved = 0;
    unsigned long long local_transferred = 0;
    double local_reconstruct_time = 0;
    unsigned long long local_num_elements = 0;
    double rel_tolerance = 0;
    std::vector<double> local_errors(verify == 1 ? num_frames : 0, 0);
    std::vector<double> max_errors(local_errors.size(), 0);
    std::vector<double> frame_ranges(verify == 1 ? num_frames : 0, 0);
    std::vector<double> frame_bounds(verify == 1 ? num_frames : 0, 0);

    for (int timestep = 0; timestep < num_timesteps; ++timestep) {
        for (size_t field = 0; field < variables.size(); ++field) {
            const size_t frame = static_cast<size_t>(timestep) * variables.size() + field;
            std::vector<uint8_t> metadata;
            std::vector<std::vector<uint8_t>> components;
            if (!archive.read_block_components(
                    IPCompFUN3D::frame_block_index(timestep, field, variables.size()),
                    metadata, components)) {
                return 1;
            }
            if (metadata.size() < sizeof(IPCompFUN3D::FrameInfo) || components.size() != 2) {
                fprintf(stderr, "ERROR: rank %d: a block of %s is not a retrieve bundle "
                                "block (FrameInfo + range table + bytes)\n",
                        rank, archive.path().c_str());
                return 1;
            }
            IPCompFUN3D::FrameInfo info;
            memcpy(&info, metadata.data(), sizeof(info));
            IPCompFUN3D::RetrievalRecord record;
            if (!IPCompFUN3D::split_record(metadata, record)) {
                fprintf(stderr, "ERROR: rank %d: a block of %s has no retrieval record; "
                                "was it written by para_IPComp_retrieve_fun3d?\n",
                        rank, archive.path().c_str());
                return 1;
            }
            rel_tolerance = record.rel_tolerance;
            local_transferred += metadata.size() + components[0].size() + components[1].size();

            std::vector<T> original;
            if (verify == 1) {
                const std::string input = original_dir + variables[field] + ".dat." +
                                          std::to_string(timestep);
                if (!IPCompFUN3D::read_local_field(input, partition_prefix, local_partition,
                                                   partitions_per_subdomain, original)) {
                    fprintf(stderr, "ERROR: rank %d cannot read the original %s\n",
                            rank, input.c_str());
                    return 1;
                }
                frame_ranges[frame] = info.value_range;
                frame_bounds[frame] = record.abs_tolerance;
            }

            std::vector<T> decoded;
            std::string error;
            if (!reconstruct_frame<T>(components[0], components[1], info,
                                      record.rel_tolerance, comm, local_retrieved,
                                      local_reconstruct_time,
                                      verify == 1 ? &decoded : nullptr, error)) {
                fprintf(stderr, "ERROR: rank %d, %s timestep %d: %s\n",
                        rank, variables[field].c_str(), timestep, error.c_str());
                return 1;
            }
            if (verify == 1) {
                local_errors[frame] = max_abs_error(original, decoded);
            }
            local_num_elements += info.num_elements;
        }
    }

    unsigned long long total_num_elements = 0;
    unsigned long long total_retrieved = 0;
    unsigned long long total_transferred = 0;
    double max_reconstruct_time = 0;
    MPI_Reduce(&local_num_elements, &total_num_elements, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_retrieved, &total_retrieved, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_transferred, &total_transferred, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_reconstruct_time, &max_reconstruct_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, comm);
    if (verify == 1) {
        MPI_Reduce(local_errors.data(), max_errors.data(),
                   static_cast<int>(local_errors.size()), MPI_DOUBLE, MPI_MAX, 0, comm);
    }
    if (rank == 0) {
        printf("IPComp-Retrieved ranks=%d preprocessing=0.000000\n", np);
        // `retrieved` is the codec's own count, as the plain reconstructor reports it;
        // `transferred` is what the bundle actually carried for these frames.
        printf("  rel_tol=%.3g retrieved=%llu bitrate=%.4f ratio=%.4f transferred=%llu "
               "reconstruct=%.6f\n",
               rel_tolerance, total_retrieved,
               total_retrieved * 8.0 / total_num_elements,
               static_cast<double>(total_num_elements) * sizeof(T) / total_retrieved,
               total_transferred, max_reconstruct_time);
        if (verify == 1) {
            int violations = 0;
            for (size_t frame = 0; frame < num_frames; ++frame) {
                const int timestep = static_cast<int>(frame / variables.size());
                const std::string& variable = variables[frame % variables.size()];
                const bool violated = !(max_errors[frame] <= frame_bounds[frame]);
                violations += violated ? 1 : 0;
                printf("  verify timestep=%d field=%s rel_tol=%.3g range=%.6g bound=%.6g "
                       "max_error=%.6g %s\n",
                       timestep, variable.c_str(), rel_tolerance, frame_ranges[frame],
                       frame_bounds[frame], max_errors[frame],
                       violated ? "VIOLATED" : "ok");
            }
            printf("  verify frames=%zu violations=%d\n", num_frames, violations);
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
