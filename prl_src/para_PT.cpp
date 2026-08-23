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
    // int max_index = 0;
	for(int i=1; i<vec.size(); i++){
		if(max < fabs(vec[i])) {
            max = fabs(vec[i]);
            // max_index = i;
        }
	}
	// std::cout << ": max absolute value = " << max << ", max_index = " << max_index << std::endl;
	return max;
}

template<class T>
bool halfing_error_PT_uniform(const T * Vx, const T * Vy, const T * Vz, const T * P, const T * D, size_t n, const std::vector<unsigned char>& mask, const double tau, std::vector<double>& ebs, std::vector<T>& PT_ori, std::vector<double>& error_est_PT, std::vector<double>& error_PT){
	double eb_Vx = ebs[0];
	double eb_Vy = ebs[1];
	double eb_Vz = ebs[2];
	double eb_P = ebs[3];
	double eb_D = ebs[4];
	double R = 287.1;
	double gamma = 1.4;
	double mi = 3.5;
	double mu_r = 1.716e-5;
	double T_r = 273.15;
	double S = 110.4;
	double c_1 = 1.0 / R;
	double c_2 = sqrt(gamma * R);
	int C7i[8] = {1, 7, 21, 35, 35, 21, 7, 1};
	double max_value = 0;
	int max_index = 0;
	int n_variable = ebs.size();
	double Mach_tmp_pow[8];
    double e_Mach_tmp_pow[8];
	for(int i=0; i<n; i++){
		double e_V_TOT_2 = 0;
		e_V_TOT_2 = mask[i] ? MDR::compute_bound_x_square<double>(Vx[i], eb_Vx) + MDR::compute_bound_x_square<double>(Vy[i], eb_Vy) + MDR::compute_bound_x_square<double>(Vz[i], eb_Vz) : 0;
		double V_TOT_2 = Vx[i]*Vx[i] + Vy[i]*Vy[i] + Vz[i]*Vz[i];
		double e_V_TOT = 0;
		e_V_TOT = mask[i] ? MDR::compute_bound_square_root_x<double>(V_TOT_2, e_V_TOT_2) : 0;
		double V_TOT = sqrt(V_TOT_2);
		double e_T = c_1 * MDR::compute_bound_division<double>(P[i], D[i], eb_P, eb_D);
		double Temp = P[i] / (D[i] * R);
		double e_C = c_2*MDR::compute_bound_square_root_x<double>(Temp, e_T);
		double C = c_2 * sqrt(Temp);
		double e_Mach = MDR::compute_bound_division<double>(V_TOT, C, e_V_TOT, e_C);
		double Mach = V_TOT / C;
		double e_Mach_tmp = ldexp(gamma - 1, -1) * MDR::compute_bound_x_square<double>(Mach, e_Mach);
		double Mach_tmp = 1 + ldexp(gamma - 1, -1) * Mach * Mach;
		double e_Mach_tmp_mi = 0;
        Mach_tmp_pow[0] = 1;
        e_Mach_tmp_pow[0] = 1;
        for (int k = 1; k <= 7; k++) {
            Mach_tmp_pow[k] = Mach_tmp_pow[k - 1] * Mach_tmp;
            e_Mach_tmp_pow[k] = e_Mach_tmp_pow[k - 1] * e_Mach_tmp;
        }
        for (int k = 1; k <= 7; k++) {
            e_Mach_tmp_mi += C7i[k] * Mach_tmp_pow[7 - k] * e_Mach_tmp_pow[k];
        }
		double Mach_tmp_mi = sqrt(Mach_tmp * Mach_tmp * Mach_tmp * Mach_tmp * Mach_tmp * Mach_tmp * Mach_tmp);
		double e_PT = MDR::compute_bound_multiplication<double>(P[i], Mach_tmp_mi, eb_P, e_Mach_tmp_mi);
		double PT = P[i] * Mach_tmp_mi;

		error_est_PT[i] = e_PT;
		error_PT[i] = PT - PT_ori[i];
		if(max_value < error_est_PT[i]){
			max_value = error_est_PT[i];
			max_index = i;
		}
	}
	if(max_value > tau){
		auto i = max_index;
		double estimate_error = max_value;
		double eb_Vx = ebs[0];
		double eb_Vy = ebs[1];
		double eb_Vz = ebs[2];
		double eb_P = ebs[3];
		double eb_D = ebs[4];
		while(estimate_error > tau){
    		// std::cout << "uniform decrease\n";
			eb_Vx = eb_Vx / 1.5;
			eb_Vy = eb_Vy / 1.5;
			eb_Vz = eb_Vz / 1.5; 
			eb_P = eb_P / 1.5;
			eb_D = eb_D / 1.5;
			double e_V_TOT_2 = 0;
			if(mask[i]) e_V_TOT_2 = MDR::compute_bound_x_square<double>(Vx[i], eb_Vx) + MDR::compute_bound_x_square<double>(Vy[i], eb_Vy) + MDR::compute_bound_x_square<double>(Vz[i], eb_Vz);
			double V_TOT_2 = Vx[i]*Vx[i] + Vy[i]*Vy[i] + Vz[i]*Vz[i];
			double e_V_TOT = 0;
			if(mask[i]) e_V_TOT = MDR::compute_bound_square_root_x<double>(V_TOT_2, e_V_TOT_2);
			double V_TOT = sqrt(V_TOT_2);
			double e_T = c_1 * MDR::compute_bound_division<double>(P[i], D[i], eb_P, eb_D);
			double Temp = P[i] / (D[i] * R);
			double e_C = c_2*MDR::compute_bound_square_root_x<double>(Temp, e_T);
			double C = c_2 * sqrt(Temp);
			double e_Mach = MDR::compute_bound_division<double>(V_TOT, C, e_V_TOT, e_C);
			double Mach = V_TOT / C;
			double e_Mach_tmp = (gamma-1) / 2 * MDR::compute_bound_x_square<double>(Mach, e_Mach);
			double Mach_tmp = 1 + (gamma-1)/2 * Mach * Mach;
			double e_Mach_tmp_mi = 0;
			Mach_tmp_pow[0] = 1;
            e_Mach_tmp_pow[0] = 1;
            for (int k = 1; k <= 7; k++) {
                Mach_tmp_pow[k] = Mach_tmp_pow[k - 1] * Mach_tmp;
                e_Mach_tmp_pow[k] = e_Mach_tmp_pow[k - 1] * e_Mach_tmp;
            }
            for (int k = 1; k <= 7; k++) {
                e_Mach_tmp_mi += C7i[k] * Mach_tmp_pow[7 - k] * e_Mach_tmp_pow[k];
            }
            double Mach_tmp_mi = sqrt(Mach_tmp * Mach_tmp * Mach_tmp * Mach_tmp * Mach_tmp * Mach_tmp * Mach_tmp);
			estimate_error = MDR::compute_bound_multiplication<double>(P[i], Mach_tmp_mi, eb_P, e_Mach_tmp_mi);
            // if((ebs[0] / eb_Vx) > 10) break;
		}
		ebs[0] = eb_Vx;
		ebs[1] = eb_Vy;
		ebs[2] = eb_Vz;
		ebs[3] = eb_P;
		ebs[4] = eb_D;
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
    std::vector<std::string> var_list = {"VelocityX", "VelocityY", "VelocityZ", "Pressure", "Density"};
    int n_variable = var_list.size();
    std::vector<std::unique_ptr<T[]>> vars_vec;
    std::vector<std::unique_ptr<SZ3::uchar[]>> vars_cmp;
    vars_vec.reserve(n_variable);
    vars_cmp.reserve(n_variable);
    std::vector<double> targetEBs;
    for(int i=0; i<n_variable; i++){
        auto original_data = SZ3::readfile<T>((data_file_prefix + var_list[i] + ".dat").c_str(), num_elements);
        // std::cout << "target_eb * compute_global_value_range(original_data.get(), num_elements) = " << target_eb * compute_global_value_range(original_data.get(), num_elements) << std::endl; 
        targetEBs.push_back(target_eb * compute_global_value_range(original_data.get(), num_elements));
        // if(!rank) std::cout << var_list[i] << " value range: " << compute_global_value_range(original_data.get(), num_elements) << std::endl;
        vars_vec.push_back(std::move(original_data));

        auto cmp_data = SZ3::readfile<SZ3::uchar>((rdata_file_prefix + var_list[i] + "_refactored/" + var_list[i] + "_psz.bin").c_str(), compressed_elements);
        vars_cmp.push_back(std::move(cmp_data));
    }

    std::vector<T> PT_ori(num_elements, 0);
    MDR::compute_PT(vars_vec[0].get(), vars_vec[1].get(), vars_vec[2].get(), vars_vec[3].get(), vars_vec[4].get(), num_elements, PT_ori.data());
    // std::cout << "PT_ori[349523] = " << PT_ori[349523] << std::endl;
    target_eb *= compute_global_value_range(PT_ori.data(), num_elements);
    // if(!rank) std::cout << "compute_global_value_range(PT_ori.data(), num_elements) = " << compute_global_value_range(PT_ori.data(), num_elements) << std::endl;

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
        sz.setupLayersFromRange(
            compute_global_value_range(vars_vec[i].get(), num_elements));
        reconstructors.push_back(sz);
    }

    int max_iter = 30, iter = 0;
    bool tolerance_met = false;
    std::vector<std::vector<T>> reconstructed_vars(n_variable, std::vector<T>(num_elements));
    std::vector<size_t> total_retrieved_size(n_variable, 0);
    std::vector<double> error_PT(num_elements);
    std::vector<double> error_est_PT(num_elements);
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
            if (i < 3){
                for(int j=0; j<num_elements; j++){
                    if(!mask[j]) reconstructed_vars[i][j] = 0;
                }
            }
        }
        T * Vx_dec = reconstructed_vars[0].data();
        T * Vy_dec = reconstructed_vars[1].data();
        T * Vz_dec = reconstructed_vars[2].data();
        T * P_dec = reconstructed_vars[3].data();
        T * D_dec = reconstructed_vars[4].data();
        tolerance_met = halfing_error_PT_uniform(Vx_dec, Vy_dec, Vz_dec, P_dec, D_dec, num_elements, mask, target_eb, targetEBs, PT_ori, error_est_PT, error_PT);
        // std::cout << "error_PT[349523] = " << error_PT[349523] << std::endl;
        max_act_error = print_max_abs(error_PT);
        // std::cout << "error_est_PT[349523] = " << error_est_PT[349523] << std::endl;
        max_est_error = print_max_abs(error_est_PT);  
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
    if(!rank) std::cout << "max_est_error = " << global_max_est_error << std::endl;
	
    double global_max_act_error = 0;
    MPI_Reduce(&max_act_error, &global_max_act_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(!rank) std::cout << "max_act_error = " << global_max_act_error << std::endl;

    unsigned long long local_total_size = std::accumulate(total_retrieved_size.begin(), total_retrieved_size.end(), 0ULL) + mask_file_size;

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
        for (int j=0; j < level_bitplane_info.size(); j++){
            for(int k=31; k >= 0; k--){
                if(31 - k < level_bitplane_info[j]){
                    memcpy(dst_ptr, src_ptr, lossless_size[1 + j*32 + k]);
                    dst_ptr += lossless_size[1 + j*32 + k];
                }
                src_ptr += lossless_size[1 + j*32 + k];
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
        if(i == 0){
            // mask file size already known
			unsigned long long int mask_offset = 0;
			unsigned long long int mask_buffer; 
			for(int j=0; j<size; j++){
				if(j == rank){
					if(j != 0) {
						MPI_Recv(&mask_offset, 1, MPI_UNSIGNED_LONG_LONG, j-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					}
					mask_buffer = mask_offset + mask_file_size;
					if(j != size - 1) {
						MPI_Send(&mask_buffer, 1, MPI_UNSIGNED_LONG_LONG, j+1, 0, MPI_COMM_WORLD);
					}
				}
			}
			size_t mask_num_char = 0;
            auto mask_data = SZ3::readfile<unsigned char>(mask_file.c_str(), mask_num_char);
			MPI_File mask_file;
			std::string mask_filename = wdata_file_prefix + "aggregated_psz_mask.bin";
			MPI_File_open(MPI_COMM_WORLD, mask_filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &mask_file);
			MPI_File_write_at(mask_file, mask_offset, mask_data.get(), mask_file_size, MPI_SIGNED_CHAR, MPI_STATUS_IGNORE);
			MPI_File_close(&mask_file);
        }
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
    else {
        std::cout << "No PT for " << data_name << " dataset." << std::endl;
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
