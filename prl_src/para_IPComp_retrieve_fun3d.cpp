// Retrieve stage of a retrieve-then-transfer workflow for IPComp.
//
// IPComp selects its layers inside the codec, so this stage cannot plan what to fetch by
// reading metadata the way the other codecs do.  Instead it runs ONE reconstruction per
// frame at the requested tolerance with the codec's access log switched on, learns
// exactly which byte ranges of the stream the codec consumed -- each layer's header
// block, the unpredictable block, and the bitgroup blocks its strategy chose -- and
// writes only those into the bundle, with their offsets, so the reconstruct stage can put
// them back where the codec expects them.  The bundle is one file per rank in the same
// archive layout, plus metadata.json beside it; it can be moved to another cluster and
// reconstructed there with para_IPComp_reconstruct_retrieved_fun3d.
//
// Because the decision IS a decode here, `retrieve=` includes decoding the frame.

#include <mpi.h>

#include <SZ3/compressor/IPComp.hpp>
#include <SZ3/encoder/BypassEncoder.hpp>
#include <SZ3/lossless/Lossless_zstd.hpp>
#include <SZ3/quantizer/IntegerQuantizer2.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "FUN3DIPComp.hpp"

// Reconstructs once with the log on and returns the merged ranges the codec touched.
// `codec_retrieved` is the codec's own accounting of the same thing.
template <class T>
bool touched_ranges(std::vector<uint8_t>& stream, const IPCompFUN3D::FrameInfo& info,
                    double rel_tolerance, std::vector<IPCompFUN3D::ByteRange>& ranges,
                    size_t& codec_retrieved) {
    using Codec = SZ3::SZProgressiveMQuant<
        T, 1, SZ3::LinearQuantizer2<T>, SZ3::BypassEncoder<int>,
        SZ3::Lossless_zstd>;
    const std::array<size_t, 1> dimensions{static_cast<size_t>(info.num_elements)};
    Codec codec(SZ3::LinearQuantizer2<T>(info.num_elements, 1),
                SZ3::BypassEncoder<int>(), SZ3::Lossless_zstd(), dimensions,
                1, 0, 50000, info.layers, 0);
    if (stream.empty()) return false;
    codec.setupLayersFromRange(static_cast<T>(info.value_range));

    std::vector<IPCompFUN3D::ByteRange> log;
    codec.set_access_log(&log);
    std::vector<double> target{rel_tolerance * info.value_range};
    codec.progressive_reconstruct(
        reinterpret_cast<SZ3::uchar*>(stream.data()), nullptr, target);
    codec.set_access_log(nullptr);
    codec_retrieved = codec.get_retrieved_size();
    codec.release_workspace();

    ranges = IPCompFUN3D::merge_ranges(std::move(log));
    for (const auto& r : ranges) {
        if (r.first + r.second > stream.size()) {
            fprintf(stderr, "ERROR: codec logged a range past the end of the stream\n");
            return false;
        }
    }
    return !ranges.empty();
}

template <class T>
int run(int argc, char** argv, MPI_Comm comm) {
    int rank = 0;
    int np = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);
    if (argc != 7) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s input_dir output_dir variables nt rel_tol f|d\n",
                    argv[0]);
        }
        return 1;
    }
    if (np < 72 || np % 72 != 0) {
        if (rank == 0) fprintf(stderr, "ERROR: ranks must be a multiple of 72\n");
        return 1;
    }

    std::string input_root = argv[1];
    std::string output_root = argv[2];
    if (input_root.back() != '/') input_root += '/';
    if (output_root.back() != '/') output_root += '/';
    const int num_timesteps = std::atoi(argv[4]);
    const double rel_tolerance = std::atof(argv[5]);
    if (num_timesteps < 1 || !(rel_tolerance > 0)) return 1;

    const int subdomain = rank % 72;
    const int local_partition = rank / 72;
    const std::string input_subdomain = IPCompFUN3D::subdomain_dir(input_root, subdomain);
    const auto variables = IPCompFUN3D::variables(input_subdomain, argv[3]);
    if (variables.empty()) return 1;
    if (!IPCompFUN3D::copy_metadata(input_subdomain, output_root, subdomain,
                                    local_partition)) {
        fprintf(stderr, "ERROR: rank %d cannot copy metadata.json into %s\n",
                rank, output_root.c_str());
        return 1;
    }

    IPCompFUN3D::SingleFileArchive archive(
        IPCompFUN3D::single_file_archive_name(input_root, np, rank));
    if (!archive.open()) return 1;
    if (archive.num_fields() != variables.size() ||
        archive.num_timesteps() < static_cast<uint64_t>(num_timesteps) ||
        archive.header().element_size != sizeof(T)) {
        fprintf(stderr, "ERROR: rank %d: %s does not match this run "
                        "(fields, timesteps or f/d)\n",
                rank, archive.path().c_str());
        return 1;
    }

    IPCompFUN3D::SingleFileWriter bundle(
        IPCompFUN3D::retrieved_archive_name(output_root, np, rank),
        static_cast<uint64_t>(num_timesteps),
        static_cast<uint64_t>(variables.size()), sizeof(T));
    if (!bundle.open()) return 1;

    unsigned long long local_bundle_bytes = bundle.prologue_size();
    unsigned long long local_payload_bytes = 0;      // ranges + tables + metadata shipped
    unsigned long long local_codec_retrieved = 0;    // the codec's own count, to compare
    unsigned long long local_full_stream = 0;        // what the whole streams would have been
    double local_retrieve_time = 0;                  // read + decode-to-learn, excludes the write

    for (int timestep = 0; timestep < num_timesteps; ++timestep) {
        for (size_t field = 0; field < variables.size(); ++field) {
            const uint64_t index =
                IPCompFUN3D::frame_block_index(timestep, field, variables.size());

            MPI_Barrier(comm);
            const double start = MPI_Wtime();
            std::vector<uint8_t> metadata;
            std::vector<uint8_t> stream;
            if (!archive.read_block(index, metadata, stream)) return 1;
            if (metadata.size() < sizeof(IPCompFUN3D::FrameInfo)) {
                fprintf(stderr, "ERROR: rank %d read a block of %s with no FrameInfo\n",
                        rank, archive.path().c_str());
                return 1;
            }
            IPCompFUN3D::FrameInfo info;
            memcpy(&info, metadata.data(), sizeof(info));

            std::vector<IPCompFUN3D::ByteRange> ranges;
            size_t codec_retrieved = 0;
            if (!touched_ranges<T>(stream, info, rel_tolerance, ranges, codec_retrieved)) {
                fprintf(stderr, "ERROR: rank %d could not learn the access pattern of "
                                "%s at timestep %d\n",
                        rank, variables[field].c_str(), timestep);
                return 1;
            }
            local_retrieve_time += MPI_Wtime() - start;

            // ---- pack: the range table, then the bytes of every range in order ----
            const auto table = IPCompFUN3D::serialize_range_table(stream.size(), ranges);
            std::vector<uint8_t> bytes;
            size_t touched = 0;
            for (const auto& r : ranges) touched += r.second;
            bytes.reserve(touched);
            for (const auto& r : ranges) {
                bytes.insert(bytes.end(), stream.begin() + r.first,
                             stream.begin() + r.first + r.second);
            }

            IPCompFUN3D::RetrievalRecord record;
            record.rel_tolerance = rel_tolerance;
            record.abs_tolerance = rel_tolerance * info.value_range;
            record.taken = {static_cast<uint64_t>(touched),
                            static_cast<uint64_t>(codec_retrieved)};
            const auto bundle_metadata = IPCompFUN3D::append_record(metadata, record);

            bundle.begin_block();
            bundle.add_component(table.data(), table.size());
            bundle.add_component(bytes.data(), bytes.size());
            size_t block_size = 0;
            if (!bundle.commit_block(index, bundle_metadata.data(), bundle_metadata.size(),
                                     block_size)) {
                return 1;
            }
            local_bundle_bytes += block_size;
            local_payload_bytes += bundle_metadata.size() + table.size() + bytes.size();
            local_codec_retrieved += codec_retrieved;
            local_full_stream += stream.size();
        }
    }
    if (!bundle.close()) return 1;

    unsigned long long total_bundle_bytes = 0;
    unsigned long long total_payload_bytes = 0;
    unsigned long long total_codec_retrieved = 0;
    unsigned long long total_full_stream = 0;
    double max_retrieve_time = 0;
    double local_read_time = archive.io_time();
    double max_read_time = 0;
    double local_write_time = bundle.io_time();
    double max_write_time = 0;
    MPI_Reduce(&local_bundle_bytes, &total_bundle_bytes, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_payload_bytes, &total_payload_bytes, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_codec_retrieved, &total_codec_retrieved, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_full_stream, &total_full_stream, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_retrieve_time, &max_retrieve_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_read_time, &max_read_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_write_time, &max_write_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("IPComp-Retrieve ranks=%d timesteps=%d fields=%zu rel_tol=%.3g "
               "retrieve=%.6f read_io=%.6f write_io=%.6f "
               "retrieved_payload=%llu codec_retrieved=%llu full_streams=%llu "
               "bundle_bytes=%llu files=%d\n",
               np, num_timesteps, variables.size(), rel_tolerance,
               max_retrieve_time, max_read_time, max_write_time,
               total_payload_bytes, total_codec_retrieved, total_full_stream,
               total_bundle_bytes, np);
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
