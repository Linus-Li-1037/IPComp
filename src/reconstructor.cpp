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

bool write_output = false;
std::string output_path;

template<uint N, typename T, class ... Dims>
T *interp_decompress(const char *path, std::vector<double> & target_ebs, int interp_op, int direction_op,
                                int layers, std::string filename, bool writeintoFile, Dims ... args){
    size_t num = 0;
    auto data = SZ3::readfile<T>(path, num);
    T * dec_data = nullptr;

    {
    std::cout << "****************** Decompression ****************" << std::endl;

    auto dims = std::array<size_t, N>{static_cast<size_t>(std::forward<Dims>(args))...};
    auto sz = SZ3::SZProgressiveMQuant<T, N, SZ3::LinearQuantizer2<T>, SZ3::BypassEncoder<int>, SZ3::Lossless_zstd>(
            // SZ3::LinearQuantizer2<T>(num, eb, 524288),
            SZ3::LinearQuantizer2<T>(num, 1), // the second arg is dummy.
            SZ3::BypassEncoder<int>(),
            // SZ3::ArithmeticEncoder<int>(),
            SZ3::Lossless_zstd(),
            dims, interp_op, direction_op, 50000, layers, 0
    );
    sz.setupLayers(data.get());
    T range = sz.get_range();
    for(auto & ebs: target_ebs){
        ebs *= range;
    }

    size_t num1 = 0;
    auto compressed = SZ3::readfile<SZ3::uchar>(filename.c_str(), num1);
    
    SZ3::Timer timer(true);

    dec_data = sz.progressive_reconstruct(compressed.get(), data.get(), target_ebs);

    timer.stop("Decompression");

    if (writeintoFile){
//        std::string file = std::string(path).substr(std::string(path).rfind('/') + 1) + ".sz3.out";
//        std::cout << "decompressed file = " << file << std::endl;
//        SZ3::writefile(file.c_str(), dec_data.get(), num);
    }

    // if (level_independent <= 0) {
    //     size_t num1 = 0;
    //     auto ori_data = SZ3::readfile<float>(path, num1);
    //     assert(num1 == num);
    //     double psnr, nrmse;
    //     SZ3::verify<float>(ori_data.get(), dec_data, num, psnr, nrmse);
    //     delete[]dec_data;
    //     delete[]compressed;
    // }
//        std::vector<float> error(num);
//        for (size_t i = 0; i < num; i++) {
//            error[i] = ori_data[i] - dec_data[i];
//        }
//        std::string error_file(path);
//        error_file += ".error";
//        SZ3::writefile(error_file.c_str(), error.data(), num);
//        auto compression_ratio = num * sizeof(float) * 1.0 / total_compressed_size;
//        printf("PSNR = %f, NRMSE = %.10G, Compression Ratio = %.2f\n", psnr, nrmse, compression_ratio);
    size_t retrieved_size = sz.get_retrieved_size();
    size_t num_elements = num;
    std::cout << "[Log] Target (absolute) error bound = " << target_ebs[0] << std::endl;
    size_t last_rs = 0;
    // timer.stop("pre decmp -4");
    // decompress(lossless_data, data, dec_data, targetEBs[0], 0);
    // timer.stop("decmp ");
    // decompress(lossless_data, data, dec_data, targetEBs[0]* 2, 0);
    // decompress(lossless_data, data, dec_data, targetEBs[0]* (1 + log2(targetEBs[0] / ebs[0]) / 16.), 0);
    std::cout << "[Log] Data Chunk #1: " << "size = " << retrieved_size << " Bytes (" << retrieved_size * 100.0 / (num_elements * sizeof(T)) << "\% original data)" << std::endl;
    printf("[Log] Total Retrieved size = %lu Bytes (%.3f%% original data, bitrate = %.3f bps)\n", retrieved_size, retrieved_size * 100.0 / (num_elements * sizeof(T)), retrieved_size * 8.0 * sizeof(T) / (num_elements * sizeof(T)) );
    printf("Compression Ratio = %.5f\n", (num_elements * sizeof(T) * 1.0) / retrieved_size);
    printf("Bitrate = %.5f\n", retrieved_size * 8.0 / num_elements);
    double psnr, nrmse;
    SZ3::verify<T>(data.get(), dec_data, num_elements, psnr, nrmse);
    printf("PSNR = %.5f\n", psnr);
    last_rs = retrieved_size;
    if(write_output) SZ3::writefile(output_path.c_str(), dec_data, num);
}
    return dec_data;
}
template<uint N, class ... Dims>
double interp_decompress(const char *path, std::string rdata_path, std::vector<double> &target_ebs, int interp_op, int direction_op,
                                int layers, const char *dataType, Dims ... args) {
    double compression_ratio = -1;
    size_t compressed_size = 0;
    std::string filename = rdata_path + "/psz.bin";
    if(dataType[0] == 'f') {
        printf("[Log] dataType: %s\n", "float");
        size_t num = 0;
        // auto compressed = SZ3::readfile<SZ3::uchar>(filename.c_str(), num);
        float * dec_data = interp_decompress<N, float>(path, target_ebs, interp_op, direction_op, layers,
                                                filename, false, std::forward<Dims>(args)...);
    } else if(dataType[0] == 'd') {
        printf("[Log] dataType: %s\n", "double");
        size_t num = 0;
        // auto compressed = SZ3::readfile<SZ3::uchar>(filename.c_str(), num);
        double * dec_data = interp_decompress<N, double>(path, target_ebs, interp_op, direction_op, layers,
                                                filename, false, std::forward<Dims>(args)...);
    }
    // } else if(dataType[0] == 'I') {
    //     SZ3::uchar * compressed = interp_compress<N, int32_t>(path, interp_op, direction_op, layers, 
    //                                             compression_ratio, std::forward<Dims>(args)...);
    //     int32_t * dec_data = interp_decompress<N, int32_t>(path, target_ebs, interp_op, direction_op, layers, 
    //                                             compressed, false, std::forward<Dims>(args)...);
    // }
    return compression_ratio;
}

void usage(char* cmd) {
    std::cout << "IPComp usage: " << cmd <<
                  " data_file -[dataType: f/d] -num_dim dim0 .. dimn -bound_num bound1 bound2 .. boundn refactored_path"
                  << std::endl
                  << "example: " << cmd <<
                  " density.d64 -d -3 256 384 384 -3 1e-2 1e-3 1e-4 (-cubic)" << std::endl;
}


int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 0;
    }
    int argv_id = 3;
    int dim = atoi(argv[argv_id++] + 1);
    assert(1 <= dim && dim <= 4);
    int argp = 4;
    std::vector<size_t> dims(dim);
    for (int i = 0; i < dim; i++) {
        dims[i] = atoi(argv[argv_id++]);
    }

    int mode = 0; // 0: error bound mode; 1: bit rate mode
    
    int target_eb_num = atoi(argv[argv_id++] + 1);
    
    std::vector<double> target_ebs(target_eb_num);
    for (int i = 0; i < target_eb_num; i++) {
        target_ebs[i] = atof(argv[argv_id++]);
    }

    std::string rdata_path = std::string(argv[argv_id++]);
    if(argv_id < argc) {
        output_path = std::string(argv[argv_id++]);
        write_output = true;
    }

    int interp_op = 1; // linear:0 cubic:1
    int direction_op = 0; // dimension high -> low
    int layers = 9;

    if((argv[2] + 1)[0] == 'f') {layers = 1;} // precision: 1e-6
    else if((argv[2] + 1)[0] == 'd') {layers = 9;} // precision: 1e-9

    if (dim == 1) {
        interp_decompress<1>(argv[1], rdata_path, target_ebs, interp_op, direction_op, layers,
                                    argv[2] + 1, dims[0]);
    } else if (dim == 2) {
        interp_decompress<2>(argv[1], rdata_path, target_ebs, interp_op, direction_op, layers,
                                    argv[2] + 1, dims[0], dims[1]);
    } else if (dim == 3) {
        interp_decompress<3>(argv[1], rdata_path, target_ebs, interp_op, direction_op, layers,
                                    argv[2] + 1, dims[0], dims[1], dims[2]);
    } else if (dim == 4) {
        interp_decompress<4>(argv[1], rdata_path, target_ebs, interp_op, direction_op, layers,
                                    argv[2] + 1, dims[0], dims[1], dims[2], dims[3]);
    }

    std::cout << std::endl;
    return 0;
}
