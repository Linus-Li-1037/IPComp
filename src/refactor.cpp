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

template<uint N, typename T, class ... Dims>
SZ3::uchar *interp_compress(const char *path, int interp_op, int direction_op,
                                int layers, double &compression_ratio, size_t &total_compressed_size, Dims ... args) {
    std::vector<size_t> compressed_size;

    total_compressed_size = 0;
    SZ3::uchar *compressed;

    size_t num = 0;
    // SZ3::Timer timer_io(true);
    auto data = SZ3::readfile<T>(path, num);
    // timer_io.stop("loading from disk");

    {
        std::cout << "****************** compression ******************" << std::endl;


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
        sz.setupLayers(data.get());
        // SZ3::Timer timer_compress(true);
        // timer_compress.start();
        compressed = sz.compress(data.get(), total_compressed_size, lossless_data);
        // timer_compress.stop("Compression");

        
        // total_compressed_size = std::accumulate(compressed_size.begin(), compressed_size.end(), (size_t) 0);
        compression_ratio = num * sizeof(T) * 1.0 / total_compressed_size;
        std::cout << "[Log] Compressed size = " << total_compressed_size << " Bytes" << std::endl;
        std::cout << "[Log] Compression ratio = " << compression_ratio << std::endl << std::endl;
    }
    return compressed;
}

template<uint N, class ... Dims>
double interp_compress(const char *path, std::string output_path, int interp_op, int direction_op,
                                int layers, const char *dataType, Dims ... args) {
    double compression_ratio = -1;
    size_t compressed_size = 0;
    std::string filename = output_path + "/psz.bin";
    SZ3::Timer timer_compress(true);
    timer_compress.start();
    if(dataType[0] == 'f') {
        printf("[Log] dataType: %s\n", "float");

        SZ3::uchar * compressed = interp_compress<N, float>(path, interp_op, direction_op, layers, 
                                                compression_ratio, compressed_size, std::forward<Dims>(args)...);
        SZ3::writefile(filename.c_str(), compressed, compressed_size);
        delete[] compressed;
    } else if(dataType[0] == 'd') {
        printf("[Log] dataType: %s\n", "double");

        SZ3::uchar * compressed = interp_compress<N, double>(path, interp_op, direction_op, layers, 
                                                compression_ratio, compressed_size, std::forward<Dims>(args)...);
        SZ3::writefile(filename.c_str(), compressed, compressed_size);
        delete[] compressed;
    }
    timer_compress.stop("Compression");
    return compression_ratio;
}

void usage(char* cmd) {
    std::cout << "IPComp usage: " << cmd <<
                  " data_file -[dataType: f/d] -num_dim dim0 .. dimn output_path"
                  << std::endl
                  << "example: " << cmd <<
                  " density.d64 -d -3 256 384 384 (-cubic)" << std::endl;
}


int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 0;
    }
    int argv_id = 3;
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

    if((argv[2] + 1)[0] == 'f') {layers = 1;} // precision: 1e-6
    else if((argv[2] + 1)[0] == 'd') {layers = 9;} // precision: 1e-9

    if (dim == 1) {
        interp_compress<1>(argv[1], output_path, interp_op, direction_op, layers,
                                    argv[2] + 1, dims[0]);
    } else if (dim == 2) {
        interp_compress<2>(argv[1], output_path, interp_op, direction_op, layers,
                                    argv[2] + 1, dims[0], dims[1]);
    } else if (dim == 3) {
        interp_compress<3>(argv[1], output_path, interp_op, direction_op, layers,
                                    argv[2] + 1, dims[0], dims[1], dims[2]);
    } else if (dim == 4) {
        interp_compress<4>(argv[1], output_path, interp_op, direction_op, layers,
                                    argv[2] + 1, dims[0], dims[1], dims[2], dims[3]);
    }

    std::cout << std::endl;
    return 0;
}
