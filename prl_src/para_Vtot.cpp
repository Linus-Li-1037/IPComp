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
#include "ompSZp_typemanager.h"
#include "ompSZp_typemanager.c"
#include "qoi_utils.hpp"
#include "mpi.h"

std::vector<unsigned char> readmask(const char *filepath, uint32_t & mask_file_size){
    FILE * file = fopen(filepath, "rb");
    if (file == nullptr){
        perror("Error opening file\n");
        return {};
    }
    fseek(file, 0, SEEK_END);
    uint32_t num_bytes = ftell(file);
    mask_file_size = num_bytes;
    rewind(file);
    uint8_t * mask_data = (uint8_t *)malloc(num_bytes);
    fread(mask_data, 1, num_bytes, file);
    fclose(file);
    uint8_t * mask_data_pos = mask_data;
    size_t num_elements;
    uint32_t ZSTD_mask_size;
    memcpy(&num_elements, mask_data_pos, sizeof(size_t));
    mask_data_pos += sizeof(size_t);
    memcpy(&ZSTD_mask_size, mask_data_pos, sizeof(uint32_t));
    mask_data_pos += sizeof(uint32_t);
    uint8_t * ZSTD_mask = (uint8_t *)malloc(ZSTD_mask_size);
    memcpy(ZSTD_mask, mask_data_pos, ZSTD_mask_size);
    mask_data_pos += ZSTD_mask_size;
    free(mask_data);
    size_t byteLength = 0;
    unsigned int bit_count = 1;
    unsigned int byte_count = bit_count / 8;
    unsigned remainder_bit = bit_count % 8;
    if (remainder_bit == 0){
        byteLength = byte_count * num_elements + 1;
    }
    else{
        size_t tmp = remainder_bit * num_elements;
        byteLength = byte_count * num_elements + (tmp - 1) / 8 + 1;
    }
    std::vector<unsigned char> compressed_mask(byteLength, 0);
    size_t dstCap = (size_t) ZSTD_getFrameContentSize(ZSTD_mask, ZSTD_mask_size);
    uint8_t * tmp_data = (uint8_t *) malloc(dstCap);
    uint32_t tmp_size = ZSTD_decompress(tmp_data, dstCap, ZSTD_mask, ZSTD_mask_size);

    if (tmp_data != nullptr){
        compressed_mask.assign(tmp_data, tmp_data + tmp_size);
        free(tmp_data);
    }
    // std::cout << "byteLength = " << byteLength << std::endl;
    std::vector<unsigned int> int_mask(num_elements, 0);
    std::vector<unsigned char> mask(num_elements, 0);
    if (compressed_mask.size() != Jiajun_extract_fixed_length_bits(compressed_mask.data(), num_elements, int_mask.data(), bit_count)){}
    for(int i=0; i<num_elements; i++){
        mask[i] = int_mask[i];
    }
    // std::string path = "/Users/wenboli/uky/test/ProDM/GE_small_reordered/decoded_mask.bin";
    // MGARD::writefile(path.c_str(), mask.data(), mask.size());
    // MGARD::writefile(path.c_str(), int_mask.data(), int_mask.size());
    return mask;
}

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
T getRange(T* data, size_t num_elements) {
    T max = data[0];
    T min = data[0];
    for (size_t i = 1; i < num_elements; i++) {
        if (max < data[i]) max = data[i];
        if (min > data[i]) min = data[i];
    }
    return max - min;
}

template <class T>
T print_max_abs(const std::vector<T>& vec){
	T max = fabs(vec[0]);
	for(int i=1; i<vec.size(); i++){
		if(max < fabs(vec[i])) max = fabs(vec[i]);
	}
	// std::cout << name << ": max absolute value = " << max << std::endl;
	return max;
}

template<class T>
bool halfing_error_V_TOT_uniform(const T * Vx, const T * Vy, const T * Vz, size_t n, const std::vector<unsigned char>& mask, const double tau, std::vector<double>& ebs, const std::vector<T>& V_TOT_ori, std::vector<double>& error_est_V_TOT, std::vector<double>& error_V_TOT){
	double eb_Vx = ebs[0];
	double eb_Vy = ebs[1];
	double eb_Vz = ebs[2];
	double max_value = 0;
	int max_index = 0;
	for(int i=0; i<n; i++){
		// error of total velocity square
		double e_V_TOT_2 = 0;
		e_V_TOT_2 = MDR::compute_bound_x_square<double>(Vx[i], eb_Vx) + MDR::compute_bound_x_square<double>(Vy[i], eb_Vy) + MDR::compute_bound_x_square<double>(Vz[i], eb_Vz);
		double V_TOT_2 = Vx[i]*Vx[i] + Vy[i]*Vy[i] + Vz[i]*Vz[i];
		// error of total velocity
		double e_V_TOT = 0;
		e_V_TOT = MDR::compute_bound_square_root_x<double>(V_TOT_2, e_V_TOT_2);
		double V_TOT = sqrt(V_TOT_2);
		// print_error("V_TOT", V_TOT, V_TOT_ori[i], e_V_TOT);

		error_est_V_TOT[i] = e_V_TOT;
		error_V_TOT[i] = V_TOT - V_TOT_ori[i];

		if(max_value < error_est_V_TOT[i]){
			max_value = error_est_V_TOT[i];
            max_index = i;
		}
	}
	// std::cout << "Vtot: max estimated error = " << max_value << ", index = " << max_index << ", e_V_TOT_2 = " << max_e_V_TOT_2 << ", VTOT_2 = " << max_V_TOT_2 << ", Vx = " << max_Vx << ", Vy = " << max_Vy << ", Vz = " << max_Vz << std::endl;
	// estimate error bound based on maximal errors
	if(max_value > tau){
		// estimate
		auto i = max_index;
		double estimate_error = max_value;
		double V_TOT_2 = Vx[i]*Vx[i] + Vy[i]*Vy[i] + Vz[i]*Vz[i];
		double V_TOT = sqrt(V_TOT_2);
		double eb_Vx = ebs[0];
		double eb_Vy = ebs[1];
		double eb_Vz = ebs[2];
		while(estimate_error > tau){
    		// std::cout << "uniform decrease\n";
    		// std::cout << "uniform decrease, eb_Vx / ebs[0] = " << eb_Vx / ebs[0] << std::endl;
			eb_Vx = eb_Vx / 1.5;
			eb_Vy = eb_Vy / 1.5;
			eb_Vz = eb_Vz / 1.5;		        		
			double e_V_TOT_2 = MDR::compute_bound_x_square<double>(Vx[i], eb_Vx) + MDR::compute_bound_x_square<double>(Vy[i], eb_Vy) + MDR::compute_bound_x_square<double>(Vz[i], eb_Vz);
			// float e_V_TOT = compute_bound_square_root_x(V_TOT_2, e_V_TOT_2);
			estimate_error = MDR::compute_bound_square_root_x<double>(V_TOT_2, e_V_TOT_2);
			// if (ebs[0] / eb_Vx > 10) break;
		}
		ebs[0] = eb_Vx;
		ebs[1] = eb_Vy;
		ebs[2] = eb_Vz;
		return false;
	}
	
	return true;
}

template<class T>
void reconstruct_GE(const std::string data_file_prefix, const std::string rdata_file_prefix,
                    const std::string wdata_file_prefix,
                    double target_eb,
                    int interp_op, int direction_op,
                    int layers, int rank, int size){
    size_t num_elements = 0;
    size_t compressed_elements = 0;
    std::vector<std::string> var_list = {"VelocityX", "VelocityY", "VelocityZ"};
    int n_variable = var_list.size();
    std::vector<std::unique_ptr<T[]>> vars_vec;
    std::vector<std::unique_ptr<SZ3::uchar[]>> vars_cmp;
    vars_vec.reserve(n_variable);
    vars_cmp.reserve(n_variable);
    std::vector<double> targetEBs;
    for(int i=0; i<n_variable; i++){
        auto original_data = SZ3::readfile<T>((data_file_prefix + var_list[i] + ".dat").c_str(), num_elements);
        targetEBs.push_back(target_eb * getRange(original_data.get(), num_elements));
        vars_vec.push_back(std::move(original_data));

        auto cmp_data = SZ3::readfile<SZ3::uchar>((rdata_file_prefix + var_list[i] + "_refactored/" + var_list[i] + "_psz.bin").c_str(), compressed_elements);
        vars_cmp.push_back(std::move(cmp_data));
    }

    std::vector<T> V_TOT_ori(num_elements, 0);
    MDR::compute_VTOT(vars_vec[0].get(), vars_vec[1].get(), vars_vec[2].get(), num_elements, V_TOT_ori.data());
    target_eb *= getRange(V_TOT_ori.data(), num_elements);

    std::string mask_file = rdata_file_prefix + "psz_mask.bin";
    uint32_t mask_file_size = 0;
    auto mask = readmask(mask_file.c_str(), mask_file_size);

    std::vector<SZ3::SZProgressiveMQuant<T, 1, SZ3::LinearQuantizer2<T>, SZ3::BypassEncoder<int>, SZ3::Lossless_zstd>> reconstructors;
    std::array<size_t, 1> dims = {num_elements};
    for(int i=0; i<n_variable; i++){
        auto sz = SZ3::SZProgressiveMQuant<T, 1, SZ3::LinearQuantizer2<T>, SZ3::BypassEncoder<int>, SZ3::Lossless_zstd>(
                // SZ3::LinearQuantizer2<T>(num, eb, 524288),
                SZ3::LinearQuantizer2<T>(num_elements, 1), // the second arg is dummy.
                SZ3::BypassEncoder<int>(),
                // SZ3::ArithmeticEncoder<int>(),
                SZ3::Lossless_zstd(),
                dims, interp_op, direction_op, 50000, layers, 0
        );
        sz.setupLayers(vars_vec[i].get());
        reconstructors.push_back(sz);
    }

    int max_iter = 30, iter = 0;
    bool tolerance_met = false;
    std::vector<std::vector<T>> reconstructed_vars(n_variable, std::vector<T>(num_elements));
    std::vector<size_t> total_retrieved_size(n_variable, 0);
    std::vector<double> error_V_TOT(num_elements);
    std::vector<double> error_est_V_TOT(num_elements);
    double max_est_error = 0, max_act_error = 0;

    SZ3::Timer timer(true);

    while((!tolerance_met) && (iter < max_iter)){
        iter ++;
        std::cout << "iter " << iter << std::endl;
        for(int i=0; i<n_variable; i++){
            std::vector<double> tmpEBs = {targetEBs[i]};
            auto reconstructed_data = reconstructors[i].progressive_reconstruct(vars_cmp[i].get(), vars_vec[i].get(), tmpEBs);
            total_retrieved_size[i] = reconstructors[i].get_retrieved_size();
            memcpy(reconstructed_vars[i].data(), reconstructed_data, num_elements*sizeof(T));
            for(int j=0; j<num_elements; j++){
                if(!mask[j]) reconstructed_vars[i][j] = 0;
            }
        }
        T * Vx_dec = reconstructed_vars[0].data();
        T * Vy_dec = reconstructed_vars[1].data();
        T * Vz_dec = reconstructed_vars[2].data();
        tolerance_met = halfing_error_V_TOT_uniform(Vx_dec, Vy_dec, Vz_dec, num_elements, mask, target_eb, targetEBs, V_TOT_ori, error_est_V_TOT, error_V_TOT);
        max_act_error = print_max_abs(error_V_TOT);
        max_est_error = print_max_abs(error_est_V_TOT);  
    }
    double elapsed_time = timer.stop();
    std::cout << "requested_error = " << target_eb << std::endl;
	std::cout << "max_est_error = " << max_est_error << std::endl;
	std::cout << "max_act_error = " << max_act_error << std::endl;
	std::cout << "iter = " << iter << std::endl;
    size_t total_size = mask_file_size + std::accumulate(total_retrieved_size.begin(), total_retrieved_size.end(), size_t(0));
	double cr = n_variable * num_elements * sizeof(T) * 1.0 / total_size;
	std::cout << "each retrieved size:";
    for(int i=0; i<n_variable; i++){
        std::cout << total_retrieved_size[i] << ", ";
    }
	std::cout << "mask_file_size = " << mask_file_size << std::endl;
    std::cout << "aggregated cr = " << cr << std::endl;
	std::cout << "bitrate = " << ((sizeof(T) * 8) / cr) << std::endl;
    std::cout << "elapsed_time = " << elapsed_time << std::endl;
    return;
}

template<class T, class ... Dims>
void reconstruct_3D(const std::string data_file_prefix, const std::string rdata_file_prefix,
                    const std::string wdata_file_prefix,
                    double target_eb,
                    int interp_op, int direction_op,
                    int layers, int rank, int size, Dims ... args){
    size_t num_elements = 0;
    size_t compressed_elements = 0;
    std::vector<std::string> var_list = {"VelocityX", "VelocityY", "VelocityZ"};
    int n_variable = var_list.size();
    std::vector<std::unique_ptr<T[]>> vars_vec;
    std::vector<std::unique_ptr<SZ3::uchar[]>> vars_cmp;
    vars_vec.reserve(n_variable);
    vars_cmp.reserve(n_variable);
    std::vector<double> targetEBs;
    for(int i=0; i<n_variable; i++){
        auto original_data = SZ3::readfile<T>((data_file_prefix + var_list[i] + ".dat").c_str(), num_elements);
        targetEBs.push_back(target_eb * compute_global_value_range(original_data.get(), num_elements));
        vars_vec.push_back(std::move(original_data));

        auto cmp_data = SZ3::readfile<SZ3::uchar>((rdata_file_prefix + var_list[i] + "_refactored/" + var_list[i] + "_psz.bin").c_str(), compressed_elements);
        vars_cmp.push_back(std::move(cmp_data));
    }

    std::vector<T> V_TOT_ori(num_elements, 0);
    MDR::compute_VTOT(vars_vec[0].get(), vars_vec[1].get(), vars_vec[2].get(), num_elements, V_TOT_ori.data());
    target_eb *= compute_global_value_range(V_TOT_ori.data(), num_elements);

    // std::string mask_file = rdata_file_prefix + "psz_mask.bin";
    // uint32_t mask_file_size = 0;
    // auto mask = readmask(mask_file.c_str(), mask_file_size);
    std::vector<unsigned char> mask(num_elements, 1);

    std::vector<SZ3::SZProgressiveMQuant<T, 3, SZ3::LinearQuantizer2<T>, SZ3::BypassEncoder<int>, SZ3::Lossless_zstd>> reconstructors;
    auto dims = std::array<size_t, 3>{static_cast<size_t>(std::forward<Dims>(args))...};
    for(int i=0; i<n_variable; i++){
        auto sz = SZ3::SZProgressiveMQuant<T, 3, SZ3::LinearQuantizer2<T>, SZ3::BypassEncoder<int>, SZ3::Lossless_zstd>(
                // SZ3::LinearQuantizer2<T>(num, eb, 524288),
                SZ3::LinearQuantizer2<T>(num_elements, 1), // the second arg is dummy.
                SZ3::BypassEncoder<int>(),
                // SZ3::ArithmeticEncoder<int>(),
                SZ3::Lossless_zstd(),
                dims, interp_op, direction_op, 50000, layers, 0
        );
        sz.setupLayers(vars_vec[i].get());
        reconstructors.push_back(sz);
    }

    int max_iter = 30, iter = 0;
    bool tolerance_met = false;
    std::vector<std::vector<T>> reconstructed_vars(n_variable, std::vector<T>(num_elements));
    std::vector<size_t> total_retrieved_size(n_variable, 0);
    std::vector<double> error_V_TOT(num_elements);
    std::vector<double> error_est_V_TOT(num_elements);
    double max_est_error = 0, max_act_error = 0;

    double local_elapsed_time;

    local_elapsed_time = -MPI_Wtime();
    while((!tolerance_met) && (iter < max_iter)){
        iter ++;
        // std::cout << "iter " << iter << std::endl;
        for(int i=0; i<n_variable; i++){
            std::vector<double> tmpEBs = {targetEBs[i]};
            auto reconstructed_data = reconstructors[i].progressive_reconstruct(vars_cmp[i].get(), vars_vec[i].get(), tmpEBs);
            total_retrieved_size[i] = reconstructors[i].get_retrieved_size();
            memcpy(reconstructed_vars[i].data(), reconstructed_data, num_elements*sizeof(T));
            for(int j=0; j<num_elements; j++){
                if(!mask[j]) reconstructed_vars[i][j] = 0;
            }
        }
        T * Vx_dec = reconstructed_vars[0].data();
        T * Vy_dec = reconstructed_vars[1].data();
        T * Vz_dec = reconstructed_vars[2].data();
        tolerance_met = halfing_error_V_TOT_uniform(Vx_dec, Vy_dec, Vz_dec, num_elements, mask, target_eb, targetEBs, V_TOT_ori, error_est_V_TOT, error_V_TOT);
        max_act_error = print_max_abs(error_V_TOT);
        max_est_error = print_max_abs(error_est_V_TOT);  
    }
    local_elapsed_time += MPI_Wtime();
    double global_elapsed_time = 0;
    MPI_Reduce(&local_elapsed_time, &global_elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int global_max_iter = 0;
    MPI_Reduce(&iter, &global_max_iter, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if(!rank) std::cout << "max_iter = " << global_max_iter << std::endl;

    if(!rank) std::cout << "requested_error = " << target_eb << std::endl;

    double global_max_est_error = 0;
    MPI_Reduce(&max_est_error, &global_max_est_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(!rank) std::cout << "max_est_error = " << max_est_error << std::endl;
	
    double global_max_act_error = 0;
    MPI_Reduce(&max_act_error, &global_max_act_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(!rank) std::cout << "max_act_error = " << max_act_error << std::endl;

	// std::cout << "iter = " << iter << std::endl;
    unsigned long long local_total_size = std::accumulate(total_retrieved_size.begin(), total_retrieved_size.end(), 0ULL);
	
    unsigned long long int global_total_num = 0;
    MPI_Reduce(&num_elements, &global_total_num, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	unsigned long long int global_total_retrieved = 0;
	MPI_Reduce(&local_total_size, &global_total_retrieved, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if(!rank) printf("Aggregated bitrate = %.10f, retrieved_size = %ld, total_num_elements = %ld\n", 8*global_total_retrieved * 1.0 / (global_total_num * n_variable), global_total_retrieved, global_total_num);
	if(!rank) printf("elapsed_time = %.6f\n", global_elapsed_time);

    for(int i=0; i<n_variable; i++){
        auto metadata_size = reconstructors[i].get_metadata_size();
        auto metadata1_size = reconstructors[i].get_metadata1_size();
        auto metadata1_offset = reconstructors[i].get_metadata1_offset();
        auto lossless_size = reconstructors[i].get_lossless_size();
        auto level_bitplane_info = reconstructors[i].get_level_bitplane_info();
        // if(!rank && !i) std::cout << "retrieved_size = " << total_retrieved_size[i] << std::endl;
        // if(!rank && !i) std::cout << "metadata_size = " << metadata_size << ", metadata1_size = " << metadata1_size << ", metadata1_offset = " << metadata1_offset << std::endl;

        unsigned char * metadata_ptr = vars_cmp[i].get();
        unsigned char * metadata1_ptr = vars_cmp[i].get() + metadata1_offset;
        unsigned char * level_bitplane_ptr = vars_cmp[i].get() + metadata_size;

        unsigned char * fetched_data = (unsigned char *) malloc(total_retrieved_size[i]);
        unsigned char * src_ptr = metadata_ptr;
        unsigned char * dst_ptr = fetched_data;
        // if(!rank && !i) std::cout << "Line: 414 memcpy(dst_ptr, src_ptr, metadata_size);" << std::endl;
        memcpy(dst_ptr, src_ptr, metadata_size);
        dst_ptr += metadata_size;
        src_ptr += metadata_size;
        // if(!rank && !i) std::cout << "dst_ptr - fetched_data = " << dst_ptr - fetched_data << std::endl;
        // if(!rank && !i) std::cout << "level_bitplane_info.size() = " << level_bitplane_info.size() << ", lossless_size.size() = " << lossless_size.size() << std::endl;
        // int used_bitplane_count = std::accumulate(level_bitplane_info.begin(), level_bitplane_info.end(), 0);
        // if(!rank && !i) std::cout << "used_bitplane_count = " << used_bitplane_count << std::endl;
        // int copied_bitplane_count = 0;
        // int level = 0;
        // int c = 0;
        // for(int j = 1; j < lossless_size.size() - 1; j++){
        //     if(c == 32){
        //         c = 0;
        //     }
        //     if(!c){
        //         std::cout << std::endl;
        //         std::cout << "level" << int(j/32) << std::endl;
        //     }
        //     std::cout << lossless_size[j] << " ";
        //     c++;
        // }
        // std::cout << std::endl;
        for (int j=0; j < level_bitplane_info.size(); j++){
            for(int k=31; k >= 0; k--){
                if(31 - k < level_bitplane_info[j]){
                    memcpy(dst_ptr, src_ptr, lossless_size[1 + j*32 + k]);
                    dst_ptr += lossless_size[1 + j*32 + k];
                    // copied_bitplane_count++;
                    // if(!rank && !i) std::cout << "copied_bitplane_count = " << copied_bitplane_count << ", Bitplane:" << j*32 + k << ", bitplane_size = " << lossless_size[1 + j*32 + k] << ", dst_ptr - fetched_data = " << dst_ptr - fetched_data << std::endl;
                }
                src_ptr += lossless_size[1 + j*32 + k];
            }
        }
        src_ptr = metadata1_ptr;
        memcpy(dst_ptr, src_ptr, metadata1_size);
        dst_ptr += metadata1_size;
        // if(!rank && !i) std::cout << "metadata1__, dst_ptr - fetched_data = " << dst_ptr - fetched_data << std::endl;
        // if(!rank && !i) std::cout << "memcpy complete" << std::endl;

        unsigned long long int fetched_data_offset = 0;
        unsigned long long int fetched_data_buffer;

        for(int j=0; j<size; j++){
            if(j == rank){
                if(j != 0) {
                    MPI_Recv(&fetched_data_offset, 1, MPI_UNSIGNED_LONG_LONG, j-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                fetched_data_buffer = fetched_data_offset + total_retrieved_size[i];

                if(j != size - 1) {
                    MPI_Send(&fetched_data_buffer, 1, MPI_UNSIGNED_LONG_LONG, j+1, 0, MPI_COMM_WORLD);
				}
            }
        }
        MPI_File fetched_data_file;
		std::string fetched_data_filename = wdata_file_prefix + var_list[i] + "_aggregated_fetched_data_psz.bin";
        MPI_File_open(MPI_COMM_WORLD, fetched_data_filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fetched_data_file);
        MPI_File_write_at(fetched_data_file, fetched_data_offset, fetched_data, total_retrieved_size[i], MPI_SIGNED_CHAR, MPI_STATUS_IGNORE);
		MPI_File_close(&fetched_data_file);
        free(fetched_data);
    }

    return;
}

template<class T>
void QoI_decompress_preprocess(const std::string data_name, const std::string data_prefix_path, const std::string output_path,
                                double target_eb, 
                                int interp_op, int direction_op,
                                int layers, int rank, int size){
    std::string data_file_prefix = data_prefix_path + "/data/";
    std::string rdata_file_prefix = data_prefix_path + "/refactor/";
    int exp = static_cast<int>(std::round(std::log10(target_eb)));
	std::string wdata_file_prefix = output_path + "/1e" + std::to_string(exp) + "/";
    // std::cout << "wdata_file_prefix = " << wdata_file_prefix << std::endl;
    if (std::strcmp(data_name.c_str(), "GE") == 0) {
        reconstruct_GE<T>(data_file_prefix, rdata_file_prefix, wdata_file_prefix, target_eb, interp_op, direction_op, layers, rank, size);
    }
    else if (std::strcmp(data_name.c_str(), "Hurricane") == 0){
        reconstruct_3D<T>(data_file_prefix, rdata_file_prefix, wdata_file_prefix, target_eb, interp_op, direction_op, layers, rank, size, 100, 500, 500);
    }
    else if (std::strcmp(data_name.c_str(), "NYX") == 0){
        reconstruct_3D<T>(data_file_prefix, rdata_file_prefix, wdata_file_prefix, target_eb, interp_op, direction_op, layers, rank, size, 512, 512, 512);
    }
    else if (std::strcmp(data_name.c_str(), "SCALE") == 0){
        reconstruct_3D<T>(data_file_prefix, rdata_file_prefix, wdata_file_prefix, target_eb, interp_op, direction_op, layers, rank, size, 98, 1200, 1200);
    }
    else if (std::strcmp(data_name.c_str(), "Miranda") == 0){
        reconstruct_3D<T>(data_file_prefix, rdata_file_prefix, wdata_file_prefix, target_eb, interp_op, direction_op, layers, rank, size, 256, 384, 384);
    }
    else if (std::strcmp(data_name.c_str(), "S3D") == 0){
        reconstruct_3D<T>(data_file_prefix, rdata_file_prefix, wdata_file_prefix, target_eb, interp_op, direction_op, layers, rank, size, 500, 500, 500);
    }
    else if (std::strcmp(data_name.c_str(), "Nek5000") == 0){
        reconstruct_3D<T>(data_file_prefix, rdata_file_prefix, wdata_file_prefix, target_eb, interp_op, direction_op, layers, rank, size, 510, 510, 510);
    }
    else if (std::strcmp(data_name.c_str(), "JHTDB_3GB") == 0){
        reconstruct_3D<T>(data_file_prefix, rdata_file_prefix, wdata_file_prefix, target_eb, interp_op, direction_op, layers, rank, size, 512, 512, 512);
    }
    else if (std::strcmp(data_name.c_str(), "JHTDB") == 0){
        reconstruct_3D<T>(data_file_prefix, rdata_file_prefix, wdata_file_prefix, target_eb, interp_op, direction_op, layers, rank, size, 128, 512, 512);
    }
    else if (std::strcmp(data_name.c_str(), "JHTDB_1024") == 0){
        reconstruct_3D<T>(data_file_prefix, rdata_file_prefix, wdata_file_prefix, target_eb, interp_op, direction_op, layers, rank, size, 256, 512, 512);
    }
    else if (std::strcmp(data_name.c_str(), "JHTDB_512") == 0){
        reconstruct_3D<T>(data_file_prefix, rdata_file_prefix, wdata_file_prefix, target_eb, interp_op, direction_op, layers, rank, size, 512, 512, 512);
    }
    return;                     
}

void usage(char* cmd) {
    std::cout << "para_Vtot usage: " << cmd <<
                  " data_name data_path - [dataType: f/d] requested_eb output_path"
                  << std::endl
                  << "example: " << cmd <<
                  " JHTDB ./dataset/JHTDB -f 0.1 PSZ/" << std::endl;
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::ostringstream oss;
    oss << rank;

    if (argc < 2) {
        usage(argv[0]);
        return 0;
    }

    int argv_id = 1;
    std::string data_name = argv[argv_id++];
    std::string data_path = argv[argv_id++];
    argv_id++; // data type
    data_path += oss.str();
    double tau = atof(argv[argv_id++]);
    std::string output_path = argv[argv_id++];


    int interp_op = 1; // linear:0 cubic:1
    int direction_op = 0; // dimension high -> low
    int layers = 9;

    if((argv[3] + 1)[0] == 'f') {
        layers = 1;
        QoI_decompress_preprocess<float>(data_name, data_path, output_path, tau, interp_op, direction_op, layers, rank, size);
    } // precision: 1e-6
    else if((argv[3] + 1)[0] == 'd') {
        layers = 9;
        QoI_decompress_preprocess<double>(data_name, data_path, output_path, tau, interp_op, direction_op, layers, rank, size);
    } // precision: 1e-9

    
    MPI_Finalize();
    return 0;
}
