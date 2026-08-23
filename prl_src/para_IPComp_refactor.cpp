/*
 * IPComp: Interpolation-based Progressive Compression for Scientific Data
 *
 * Copyright (c) 2025, Your Name or Your Organization
 * All rights reserved.
 *
 * This software is based on SZ (Version 3.0), developed at Argonne National Laboratory.
 * Original Copyright © 2016, UChicago Argonne, LLC
 * Authors: Sheng Di, Kai Zhao, Xin Liang, Dingwen Tao, Franck Cappello
 *
 * Licensed under the BSD 3-Clause License. See LICENSE file for details.
 */


//#include <compressor/SZProgressiveIndependentBlock.hpp>
//#include <compressor/SZProgressive.hpp>
#include <SZ3/compressor/IPComp.hpp>
#include <SZ3/quantizer/IntegerQuantizer2.hpp>
// #include <SZ3/predictor/ComposedPredictor.hpp>
#include <SZ3/lossless/Lossless_zstd.hpp>
#include <SZ3/encoder/BypassEncoder.hpp>
#include <SZ3/utils/Iterator.hpp>
#include <SZ3/utils/Verification.hpp>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <memory>
#include <type_traits>
#include <sstream>
#include "mpi.h"

template<class T>
T compute_global_value_range(const T* data, size_t n) {
    T local_min = data[0], local_max = data[0];
    for (size_t i = 1; i < n; ++i) {
        if (data[i] < local_min) local_min = data[i];
        if (data[i] > local_max) local_max = data[i];
    }
    T global_min = 0, global_max = 0;
    const MPI_Datatype type = std::is_same<T, double>::value ? MPI_DOUBLE : MPI_FLOAT;
    MPI_Allreduce(&local_min, &global_min, 1, type, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, type, MPI_MAX, MPI_COMM_WORLD);
    return global_max - global_min;
}

template<uint N, typename T, class ... Dims>
double interp_compress(std::string data_path, std::string output_path, int interp_op, int direction_op,
                                int layers, double &compression_ratio, size_t &total_compressed_size, Dims ... args) {
    std::vector<size_t> compressed_size;
    double compress_time = 0;
    total_compressed_size = 0;
    SZ3::uchar *compressed;

    size_t num = 0;
    // SZ3::Timer timer_io(true);
    auto data = SZ3::readfile<T>(data_path.c_str(), num);
    // timer_io.stop("loading from disk");
    std::string filename = output_path + "/psz.bin";

    {
        // std::cout << "****************** compression ******************" << std::endl;


        auto dims = std::array<size_t, N>{static_cast<size_t>(std::forward<Dims>(args))...};

        auto sz = SZ3::SZProgressiveMQuant<T, N, SZ3::LinearQuantizer2<T>, SZ3::BypassEncoder<int>, SZ3::Lossless_zstd>(
                // SZ3::LinearQuantizer2<T>(num, eb, 524288),
                SZ3::LinearQuantizer2<T>(num, 1), // the second arg is dummy.
                SZ3::BypassEncoder<int>(),
                // SZ3::ArithmeticEncoder<int>(),
                SZ3::Lossless_zstd(3),
                dims, interp_op, direction_op, 50000, layers, 0
        );
        SZ3::uchar *lossless_data = new SZ3::uchar[size_t((sz.num_elements < 1000000 ? 100 : 2.0) * sz.num_elements) * sizeof(T)]; //?
        sz.setupLayersFromRange(compute_global_value_range(data.get(), num));
        // SZ3::Timer timer_compress(true);
        // timer_compress.start();

        compress_time = -MPI_Wtime();
        compressed = sz.compress(data.get(), total_compressed_size, lossless_data);
        SZ3::writefile(filename.c_str(), compressed, total_compressed_size);
        compress_time += MPI_Wtime();
    }
    return compress_time;
}

template<uint N, class ... Dims>
double interp_compress(std::string path, std::string output_path, int interp_op, int direction_op,
                                int layers, const char *dataType, Dims ... args) {
    double compression_ratio = -1;
    size_t compressed_size = 0;
    double compress_time;
    if(dataType[0] == 'f') {
        // printf("[Log] dataType: %s\n", "float");
        compress_time = interp_compress<N, float>(path, output_path, interp_op, direction_op, layers, 
                                  compression_ratio, compressed_size, std::forward<Dims>(args)...);
    } else if(dataType[0] == 'd') {
        // printf("[Log] dataType: %s\n", "double");
        compress_time = interp_compress<N, double>(path, output_path, interp_op, direction_op, layers, 
                                   compression_ratio, compressed_size, std::forward<Dims>(args)...);
    }
    return compress_time;
}

void usage(char* cmd) {
    std::cout << "IPComp usage: " << std::endl <<
                  "mpirun -n #num_cores para_IPComp_refactor data_file -[dataType: f/d] -num_dim dim0 .. dimn output_path"
                  << std::endl
                  << "example: " << cmd <<
                  "mpirun -n 2 /path/density -d -3 256 384 384 /path/output" << std::endl;
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if(!rank) usage(argv[0]);
        return 0;
    }

    int argv_id = 1;
    std::string data_path = std::string(argv[argv_id++]);
    char * data_type = (argv[argv_id++] + 1);
    int dim = atoi(argv[argv_id++] + 1);
    assert(1 <= dim && dim <= 4);
    std::vector<size_t> dims(dim);
    for (int i = 0; i < dim; i++) {
        dims[i] = atoi(argv[argv_id++]);
    }
    std::string output_path = std::string(argv[argv_id++]);

    int interp_op = 1; // linear:0 cubic:1
    int direction_op = 0; // dimension high -> low
    
    int layers = 9;

    // int tmp_rank = rank;
    // int idx = tmp_rank / 64;
    // tmp_rank = tmp_rank % 64;
    // int idy = tmp_rank / 8;
    // tmp_rank = tmp_rank % 8;
    // int idz = tmp_rank;
    // data_path = data_path + std::to_string(idx) + "_" + std::to_string(idy) + "_" + std::to_string(idz) + ".d64";
    // output_path = output_path + "/" + std::to_string(idx) + "_" + std::to_string(idy) + "_" + std::to_string(idz) + "/";

    data_path = data_path + std::to_string(rank) + ".d64";
    output_path = output_path + "/" + std::to_string(rank) + "/";

    if(*data_type == 'f') {layers = 1;} // precision: 1e-6
    else if(*data_type == 'd') {layers = 9;} // precision: 1e-9

    double local_elapsed_time = 0;
    if (dim == 1) {
        local_elapsed_time = interp_compress<1>(data_path, output_path, interp_op, direction_op, layers,
                                    data_type, dims[0]);
    } else if (dim == 2) {
        local_elapsed_time = interp_compress<2>(data_path, output_path, interp_op, direction_op, layers,
                                    data_type, dims[0], dims[1]);
    } else if (dim == 3) {
        local_elapsed_time = interp_compress<3>(data_path, output_path, interp_op, direction_op, layers,
                                    data_type, dims[0], dims[1], dims[2]);
    } else if (dim == 4) {
        local_elapsed_time = interp_compress<4>(data_path, output_path, interp_op, direction_op, layers,
                                    data_type, dims[0], dims[1], dims[2], dims[3]);
    }
    double global_elapsed_time = 0;
    MPI_Reduce(&local_elapsed_time, &global_elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(!rank) std::cout << "max_elapsed_time = " << global_elapsed_time << std::endl;

    MPI_Finalize();
    return 0;
}
