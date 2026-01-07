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
bool halfing_error_XiXj_uniform(const T * Xi, const T * Xj, size_t n, const double tau, std::vector<double>& ebs, const std::vector<T>& XiXj_ori, std::vector<double>& error_est_XiXj, std::vector<double>& error_XiXj){
	double eb_Xi = ebs[0];
	double eb_Xj = ebs[1];
	double max_value = 0;
	int max_index = 0;
	int n_variable = ebs.size();
	for(int i=0; i<n; i++){
		double e_XiXj = MDR::compute_bound_multiplication<double>(Xi[i], Xj[i], eb_Xi, eb_Xj);
        double XiXj = Xi[i] * Xj[i];

		error_est_XiXj[i] = e_XiXj;
		error_XiXj[i] = XiXj - XiXj_ori[i];
		if(max_value < error_est_XiXj[i]){
			max_value = error_est_XiXj[i];
			max_index = i;
		}
	}
	// std::cout << "XiXj: max estimated error = " << max_value << ", index = " << max_index << std::endl;
	// estimate error bound based on maximal errors
	if(max_value > tau){
		auto i = max_index;
		double estimate_error = max_value;
		double eb_Xi = ebs[0];
		double eb_Xj = ebs[1];
		while(estimate_error > tau){
    		// std::cout << "uniform decrease\n";
			eb_Xi = eb_Xi / 1.5;
			eb_Xj = eb_Xj / 1.5;
			estimate_error = MDR::compute_bound_multiplication<double>(Xi[i], Xj[i], eb_Xi, eb_Xj);
		}
		ebs[0] = eb_Xi;
		ebs[1] = eb_Xj;
		return false;
	}
	return true;
}

template<class T, class ... Dims>
void reconstruct_S3D(const std::string data_file_prefix, const std::string rdata_file_prefix,
                    std::vector<int> index, double target_eb,
                    int interp_op, int direction_op,
                    int layers, Dims ... args){
    size_t num_elements = 0;
    size_t compressed_elements = 0;
    std::vector<std::string> species = {"H2", "O2", "H2O", "H", "O", "OH"};
    int n_variable = index.size();
    std::vector<std::unique_ptr<T[]>> vars_vec;
    std::vector<std::unique_ptr<SZ3::uchar[]>> vars_cmp;
    vars_vec.reserve(n_variable);
    vars_cmp.reserve(n_variable);
    std::vector<double> targetEBs;
    for(int i=0; i<n_variable; i++){
        auto original_data = SZ3::readfile<T>((data_file_prefix + species[index[i]] + ".dat").c_str(), num_elements);
        targetEBs.push_back(target_eb * getRange(original_data.get(), num_elements));
        vars_vec.push_back(std::move(original_data));

        auto cmp_data = SZ3::readfile<SZ3::uchar>((rdata_file_prefix + species[index[i]] + "_refactored/" + species[index[i]] + "_psz.bin").c_str(), compressed_elements);
        vars_cmp.push_back(std::move(cmp_data));
    }
    
    std::vector<T> XiXj(num_elements);
	for(int i=0; i<num_elements; i++){
		XiXj[i] = vars_vec[0][i] * vars_vec[1][i];
	}
    target_eb *= getRange(XiXj.data(), num_elements);

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
    std::vector<double> error_XiXj(num_elements);
    std::vector<double> error_est_XiXj(num_elements);
    double max_est_error = 0, max_act_error = 0;

    SZ3::Timer timer(true);

    while((!tolerance_met) && (iter < max_iter)){
        iter ++;
        // std::cout << "iter " << iter << std::endl;
        for(int i=0; i<n_variable; i++){
            std::vector<double> tmpEBs = {targetEBs[i]};
            auto reconstructed_data = reconstructors[i].progressive_reconstruct(vars_cmp[i].get(), vars_vec[i].get(), tmpEBs);
            total_retrieved_size[i] = reconstructors[i].get_retrieved_size();
            memcpy(reconstructed_vars[i].data(), reconstructed_data, num_elements*sizeof(T));
        }
        T * Xi_dec = reconstructed_vars[0].data();
        T * Xj_dec = reconstructed_vars[1].data();
        tolerance_met = halfing_error_XiXj_uniform(Xi_dec, Xj_dec, num_elements, target_eb, targetEBs, XiXj, error_est_XiXj, error_XiXj);
        max_act_error = print_max_abs(error_XiXj);
        max_est_error = print_max_abs(error_est_XiXj);  
    }
    double elapsed_time = timer.stop();
    std::cout << "requested_error = " << target_eb << std::endl;
	std::cout << "max_est_error = " << max_est_error << std::endl;
	std::cout << "max_act_error = " << max_act_error << std::endl;
	std::cout << "iter = " << iter << std::endl;
    size_t total_size = std::accumulate(total_retrieved_size.begin(), total_retrieved_size.end(), size_t(0));
	double cr = n_variable * num_elements * sizeof(T) * 1.0 / total_size;
	std::cout << "each retrieved size:";
    for(int i=0; i<n_variable; i++){
        std::cout << total_retrieved_size[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "aggregated cr = " << cr << std::endl;
	std::cout << "bitrate = " << ((sizeof(T) * 8) / cr) << std::endl;
    std::cout << "elapsed_time = " << elapsed_time << std::endl;
    return;
}

template<class T>
void QoI_decompress_preprocess(const std::string data_name, const std::string data_prefix_path,
                                std::vector<int> index, double target_eb,
                                int interp_op, int direction_op,
                                int layers){
    std::string data_file_prefix = data_prefix_path + "/data/";
    std::string rdata_file_prefix = data_prefix_path + "/refactor/";
    if (std::strcmp(data_name.c_str(), "S3D") == 0) {
        reconstruct_S3D<T>(data_file_prefix, rdata_file_prefix, index, target_eb, interp_op, direction_op, layers, 1200, 334, 200);
    }
    else {
        std::cout << "No XiXj for " << data_name << " dataset." << std::endl;
    }
    return;                     
}

void usage(char* cmd) {
    std::cout << "halfing_xixj usage: " << cmd <<
                  " data_name data_path - [dataType: f/d] requested_eb xi xj"
                  << std::endl
                  << "example: " << cmd <<
                  " S3D ./dataset/S3D/ -d 0.1 1 3" << std::endl;
}


int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 0;
    }

    int argv_id = 1;
    std::string data_name = argv[argv_id++];
    std::string data_path = argv[argv_id++];
    double tau = atof(argv[4]);
    int id_i = atoi(argv[5]);
    int id_j = atoi(argv[6]);
    if(!(id_i == 1 && id_j == 3) && 
        !(id_i == 4 && id_j == 5) && 
        !(id_i == 0 && id_j == 4) && 
        !(id_i == 3 && id_j == 5)){
        perror("No such QoI\n");
    }
    std::vector<int> index = {id_i, id_j};

    int interp_op = 1; // linear:0 cubic:1
    int direction_op = 0; // dimension high -> low
    int layers = 9;

    if((argv[3] + 1)[0] == 'f') {
        layers = 1;
        QoI_decompress_preprocess<float>(data_name, data_path, index, tau, interp_op, direction_op, layers);
    } // precision: 1e-6
    else if((argv[3] + 1)[0] == 'd') {
        layers = 9;
        QoI_decompress_preprocess<double>(data_name, data_path, index, tau, interp_op, direction_op, layers);
    } // precision: 1e-9

    
    std::cout << std::endl;
    return 0;
}
