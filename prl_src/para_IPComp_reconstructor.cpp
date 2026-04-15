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
T compute_global_value_range(const T * data_vec, size_t n){
	T global_max = 0, global_min = 0;
	T local_max = -std::numeric_limits<T>::max();
	T local_min = std::numeric_limits<T>::max();
	for(int i=0; i<n; i++){
		if(data_vec[i] > local_max) local_max = data_vec[i];
		if(data_vec[i] < local_min)	local_min = data_vec[i];
	}
	if(std::is_same<T, double>::value){
		MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
		MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	}
	else if(std::is_same<T, float>::value){
		MPI_Allreduce(&local_min, &global_min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
		MPI_Allreduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
	}
	return global_max - global_min;
}

template<class T>
double compute_max_abs_error(const T* ori_data, const T * rec_data, size_t size){
    double max_abs_error = 0;
    for(int i=0; i<size; i++){
        double abs_error = abs(ori_data[i] - rec_data[i]);
        if(abs_error > max_abs_error) max_abs_error = abs_error;
    }
    return max_abs_error;
}

template<uint N, typename T, class ... Dims>
void interp_decompress(std::string path, std::string compressed_path, std::string output_path, std::vector<double> & target_ebs, int interp_op, int direction_op,
                                int layers, bool writeintoFile, Dims ... args){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t num = 0;
    auto data = SZ3::readfile<T>(path.c_str(), num);
    T value_range = compute_global_value_range(data.get(), num);
    int exp = static_cast<int>(std::round(std::log10(target_ebs[0])));
    std::string wdata_file = output_path + "/1e" + std::to_string(exp) + ".bin";
    for(int i=0; i<target_ebs.size(); i++) target_ebs[i] *= value_range;
    T * dec_data = nullptr;

    {
    // std::cout << "****************** Decompression ****************" << std::endl;

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

    size_t compressed_num = 0;
    auto compressed = SZ3::readfile<SZ3::uchar>(compressed_path.c_str(), compressed_num);

    double local_elapsed_time = -MPI_Wtime();
    dec_data = sz.progressive_reconstruct(compressed.get(), data.get(), target_ebs);
    local_elapsed_time += MPI_Wtime();

    double global_elapsed_time = 0;
    MPI_Reduce(&local_elapsed_time, &global_elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(!rank) std::cout << "max_elapsed_time = " << global_elapsed_time << std::endl;

    if(!rank) std::cout << "Requested tolerance = " << target_ebs[0] << std::endl;

    double local_max_abs_error = compute_max_abs_error(data.get(), dec_data, num);
    double global_max_abs_error = 0;
    MPI_Reduce(&local_max_abs_error, &global_max_abs_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(!rank) std::cout << "max_abs_error = " << global_max_abs_error << std::endl;

    auto metadata_size = sz.get_metadata_size();
    auto metadata1_size = sz.get_metadata1_size();
    auto metadata1_offset = sz.get_metadata1_offset();
    auto lossless_size = sz.get_lossless_size();
    auto level_bitplane_info = sz.get_level_bitplane_info();
    auto retrieved_size = sz.get_retrieved_size();

    unsigned long long local_retrieved_size = static_cast<unsigned long long>(retrieved_size);
    unsigned long long global_retrieved_size = 0;
    MPI_Reduce(&local_retrieved_size, &global_retrieved_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    unsigned long long local_num_element = static_cast<unsigned long long>(num);
    unsigned long long global_num_element = 0;
    MPI_Reduce(&local_num_element, &global_num_element, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if(!rank) std::cout << "Aggregated bitrate = " << global_retrieved_size * 8.0 / global_num_element << std::endl;

    unsigned char * fetched_data = (unsigned char *) malloc(retrieved_size);
    unsigned char * metadata_ptr = compressed.get();
    unsigned char * metadata1_ptr = metadata_ptr + metadata1_offset;
    unsigned char * level_bitplane_ptr = metadata_ptr + metadata_size;
    unsigned char * src_ptr = metadata_ptr;
    unsigned char * dst_ptr = fetched_data;
    memcpy(dst_ptr, src_ptr, metadata_size);
    dst_ptr += metadata_size;
    src_ptr += metadata_size;
    for(int i=0; i < level_bitplane_info.size(); i++){
        for(int j=31; j>=0; j--){
            if(31 - j < level_bitplane_info[i]){
                memcpy(dst_ptr, src_ptr, lossless_size[1 + i*32 + j]);
                dst_ptr += lossless_size[1 + i*32 +j];
            }
            src_ptr += lossless_size[1 + i*32 + j];
        }
    }
    src_ptr = metadata1_ptr;
    memcpy(dst_ptr, src_ptr, metadata1_size);
    dst_ptr += metadata1_size;
    unsigned long long int fetched_data_offset = 0;
    unsigned long long int fetched_data_buffer;

    for(int j=0; j<size; j++){
        if(j == rank){
            if(j != 0) {
                MPI_Recv(&fetched_data_offset, 1, MPI_UNSIGNED_LONG_LONG, j-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            fetched_data_buffer = fetched_data_offset + local_retrieved_size;

            if(j != size - 1) {
                MPI_Send(&fetched_data_buffer, 1, MPI_UNSIGNED_LONG_LONG, j+1, 0, MPI_COMM_WORLD);
            }
        }
    }
    MPI_File fetched_data_file;
    MPI_File_open(MPI_COMM_WORLD, wdata_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fetched_data_file);
    MPI_File_write_at(fetched_data_file, fetched_data_offset, fetched_data, local_retrieved_size, MPI_SIGNED_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&fetched_data_file);
    free(fetched_data);
    }
    return;
}
template<uint N, class ... Dims>
void interp_decompress(std::string path, std::string rdata_path, std::string output_path, std::vector<double> &target_ebs, int interp_op, int direction_op,
                                int layers, const char *dataType, Dims ... args) {
    double compression_ratio = -1;
    size_t compressed_size = 0;
    if(dataType[0] == 'f') {
        // printf("[Log] dataType: %s\n", "float");
        interp_decompress<N, float>(path, rdata_path, output_path, target_ebs, interp_op, direction_op, layers,
                                                false, std::forward<Dims>(args)...);
    } else if(dataType[0] == 'd') {
        // printf("[Log] dataType: %s\n", "double");
        interp_decompress<N, double>(path, rdata_path, output_path, target_ebs, interp_op, direction_op, layers,
                                                false, std::forward<Dims>(args)...);
    }
    // } else if(dataType[0] == 'I') {
    //     SZ3::uchar * compressed = interp_compress<N, int32_t>(path, interp_op, direction_op, layers, 
    //                                             compression_ratio, std::forward<Dims>(args)...);
    //     int32_t * dec_data = interp_decompress<N, int32_t>(path, target_ebs, interp_op, direction_op, layers, 
    //                                             compressed, false, std::forward<Dims>(args)...);
    // }
    return;
}

void usage(char* cmd) {
    std::cout << "IPComp usage: " << std::endl <<
                  "mpirun -n #num_cores para_IPComp_reconstructor data_file -[dataType: f/d] -num_dim dim0 .. dimn -1 bound refactored_path output_path"
                  << std::endl
                  << "example: " << std::endl <<
                  "mpirun -n 2 /path/density -d -3 256 384 384 -1 1e-4 /path/compressed /path/putput" << std::endl;
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
    // rdata_path = rdata_path + "/" + std::to_string(idx) + "_" + std::to_string(idy) + "_" + std::to_string(idz) + "/psz.bin";

    data_path = data_path + std::to_string(rank) + ".d64";
    rdata_path = rdata_path + "/" + std::to_string(rank) + "/psz.bin";

    if(*data_type == 'f') {layers = 1;} // precision: 1e-6
    else if(*data_type == 'd') {layers = 9;} // precision: 1e-9

    if (dim == 1) {
        interp_decompress<1>(data_path, rdata_path, output_path, target_ebs, interp_op, direction_op, layers,
                                    data_type, dims[0]);
    } else if (dim == 2) {
        interp_decompress<2>(data_path, rdata_path, output_path, target_ebs, interp_op, direction_op, layers,
                                    data_type, dims[0], dims[1]);
    } else if (dim == 3) {
        interp_decompress<3>(data_path, rdata_path, output_path, target_ebs, interp_op, direction_op, layers,
                                    data_type, dims[0], dims[1], dims[2]);
    } else if (dim == 4) {
        interp_decompress<4>(data_path, rdata_path, output_path, target_ebs, interp_op, direction_op, layers,
                                    data_type, dims[0], dims[1], dims[2], dims[3]);
    }

    // std::cout << std::endl;
    return 0;
}
