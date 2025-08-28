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

template <class T>
void writemask(const char *filepath, T *data, size_t num_elements) {
    unsigned int bit_count = 1;
    unsigned int byte_count = bit_count / 8;
    unsigned int remainder_bit = bit_count % 8;
    size_t byteLength = 0;
    if (remainder_bit == 0) {
        byteLength = byte_count * num_elements + 1;
    } 
    else {
        size_t tmp = remainder_bit * num_elements;
        byteLength = byte_count * num_elements + (tmp - 1) / 8 + 1;
    }
    std::vector<unsigned int> int_mask(num_elements, 0);
    for(int i=0; i<num_elements; i++){
        int_mask[i] = data[i];
    }
    std::vector<unsigned char> compressed_mask(byteLength, 0);
    if (byteLength != Jiajun_save_fixed_length_bits(int_mask.data(), num_elements, compressed_mask.data(), bit_count)){}
    // uint32_t ZSTD_mask_size = ZSTD::compress(compressed_mask.data(), byteLength, &ZSTD_mask);
    size_t dst_size = ZSTD_compressBound(byteLength);
    uint8_t * ZSTD_mask = (uint8_t *) malloc(dst_size);
    uint32_t ZSTD_mask_size = ZSTD_compress(ZSTD_mask, dst_size, compressed_mask.data(), byteLength, 3);
    uint32_t mask_size = sizeof(size_t) + sizeof(uint32_t) + ZSTD_mask_size;
    uint8_t * mask_data = (uint8_t *) malloc(mask_size);
    uint8_t * mask_data_pos = mask_data;
    memcpy(mask_data_pos, &num_elements, sizeof(size_t));
    mask_data_pos += sizeof(size_t);
    memcpy(mask_data_pos, &ZSTD_mask_size, sizeof(uint32_t));
    mask_data_pos += sizeof(uint32_t);
    memcpy(mask_data_pos, ZSTD_mask, ZSTD_mask_size);
    FILE * file = fopen(filepath, "wb");
    if (file == nullptr) {
        perror("Error opening file");
        return;
    }
    fwrite(mask_data, 1, mask_size, file);
    fclose(file);
    free(mask_data);
    free(ZSTD_mask);
}

template<uint N, typename T, class ... Dims>
SZ3::uchar *interp_compress(std::unique_ptr<T[]>& data, int interp_op, int direction_op,
                                int layers, double &compression_ratio, size_t &total_compressed_size, Dims ... args) {
    std::vector<size_t> compressed_size;

    total_compressed_size = 0;
    SZ3::uchar *compressed;

    SZ3::Timer timer_io(true);
    // auto data = SZ3::readfile<T>(path, num);
    // timer_io.stop("loading from disk");

    {
        std::cout << "****************** compression ******************" << std::endl;


        auto dims = std::array<size_t, N>{static_cast<size_t>(std::forward<Dims>(args))...};
        size_t num = std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<size_t>());
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
        SZ3::Timer timer_compress(true);
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

template<class T>
void refactor_GE(const std::string data_file_prefix, const std::string rdata_file_prefix,
                    int interp_op, int direction_op,
                    int layers){
    size_t num_elements = 0;
    auto velocityX_vec = SZ3::readfile<T>((data_file_prefix + "VelocityX.dat").c_str(), num_elements);
    auto velocityY_vec = SZ3::readfile<T>((data_file_prefix + "VelocityY.dat").c_str(), num_elements);
    auto velocityZ_vec = SZ3::readfile<T>((data_file_prefix + "VelocityZ.dat").c_str(), num_elements);
    auto pressure_vec = SZ3::readfile<T>((data_file_prefix + "Pressure.dat").c_str(), num_elements);
    auto density_vec = SZ3::readfile<T>((data_file_prefix + "Density.dat").c_str(), num_elements); 
    std::vector<std::string> var_list = {"VelocityX", "VelocityY", "VelocityZ", "Pressure", "Density"};
    int n_variable = var_list.size();

    std::vector<unsigned char> mask(num_elements, 0);
    for(int i=0; i<num_elements; i++){
        // T V_total = velocityX_vec[i]*velocityX_vec[i] + velocityY_vec[i]*velocityY_vec[i] + velocityZ_vec[i]*velocityZ_vec[i];
        // std::cout << "#" << i << " V_total = " << V_total << std::endl;
        if((velocityX_vec[i]*velocityX_vec[i] + velocityY_vec[i]*velocityY_vec[i] + velocityZ_vec[i]*velocityZ_vec[i]) != 0){            
            mask[i] = 1;
        }
    }

    std::vector<std::unique_ptr<T[]>> vars_vec;
    vars_vec.reserve(n_variable);
    vars_vec.push_back(std::move(velocityX_vec));
    vars_vec.push_back(std::move(velocityY_vec));
    vars_vec.push_back(std::move(velocityZ_vec));
    vars_vec.push_back(std::move(pressure_vec));
    vars_vec.push_back(std::move(density_vec));

    std::string mask_file = rdata_file_prefix + "psz_mask.bin";
    writemask(mask_file.c_str(), mask.data(), mask.size());

    double compression_ratio = -1;
    size_t compressed_size = 0;
    std::string filename;
    for(int i=0; i<n_variable; i++){
        SZ3::uchar * compressed = interp_compress<1, T>(vars_vec[i], interp_op, direction_op, layers, compression_ratio, compressed_size, num_elements);
        filename = rdata_file_prefix + var_list[i] + "_refactored/" + var_list[i] + "_psz.bin";
        SZ3::writefile(filename.c_str(), compressed, compressed_size);
    }
    return;
}

template<class T, class ... Dims>
void refactor_3D(const std::string data_file_prefix, const std::string rdata_file_prefix,
                    int interp_op, int direction_op,
                    int layers, Dims ... args){
    size_t num_elements = 0;
    auto velocityX_vec = SZ3::readfile<T>((data_file_prefix + "VelocityX.dat").c_str(), num_elements);
    auto velocityY_vec = SZ3::readfile<T>((data_file_prefix + "VelocityY.dat").c_str(), num_elements);
    auto velocityZ_vec = SZ3::readfile<T>((data_file_prefix + "VelocityZ.dat").c_str(), num_elements);
    std::vector<std::string> var_list = {"VelocityX", "VelocityY", "VelocityZ"};
    int n_variable = var_list.size();

    std::vector<unsigned char> mask(num_elements, 0);
    for(int i=0; i<num_elements; i++){
        if((velocityX_vec[i]*velocityX_vec[i] + velocityY_vec[i]*velocityY_vec[i] + velocityZ_vec[i]*velocityZ_vec[i]) != 0){            
            mask[i] = 1;
        }
    }

    std::vector<std::unique_ptr<T[]>> vars_vec;
    vars_vec.reserve(n_variable);
    vars_vec.push_back(std::move(velocityX_vec));
    vars_vec.push_back(std::move(velocityY_vec));
    vars_vec.push_back(std::move(velocityZ_vec));

    std::string mask_file = rdata_file_prefix + "psz_mask.bin";
    writemask(mask_file.c_str(), mask.data(), mask.size());

    double compression_ratio = -1;
    size_t compressed_size = 0;
    std::string filename;
    for(int i=0; i<n_variable; i++){
        SZ3::uchar * compressed = interp_compress<3, T>(vars_vec[i], interp_op, direction_op, layers, compression_ratio, compressed_size, std::forward<Dims>(args)...);
        filename = rdata_file_prefix + var_list[i] + "_refactored/" + var_list[i] + "_psz.bin";
        SZ3::writefile(filename.c_str(), compressed, compressed_size);
    }
    return;
}

template<class T>
void QoI_compress_preprocess(const std::string data_name, const std::string data_prefix_path, 
                                int interp_op, int direction_op,
                                int layers){
    std::string data_file_prefix = data_prefix_path + "/data/";
    std::string rdata_file_prefix = data_prefix_path + "/refactor/";
    if (std::strcmp(data_name.c_str(), "GE") == 0) {
        refactor_GE<T>(data_file_prefix, rdata_file_prefix, interp_op, direction_op, layers);
    }
    else if (std::strcmp(data_name.c_str(), "Hurricane") == 0){
        refactor_3D<T>(data_file_prefix, rdata_file_prefix, interp_op, direction_op, layers, 100, 500, 500);
    }
    else if (std::strcmp(data_name.c_str(), "NYX") == 0){
        refactor_3D<T>(data_file_prefix, rdata_file_prefix, interp_op, direction_op, layers, 512, 512, 512);
    }
    else if (std::strcmp(data_name.c_str(), "SCALE") == 0){
        refactor_3D<T>(data_file_prefix, rdata_file_prefix, interp_op, direction_op, layers, 98, 1200, 1200);
    }
    else if (std::strcmp(data_name.c_str(), "Miranda") == 0){
        refactor_3D<T>(data_file_prefix, rdata_file_prefix, interp_op, direction_op, layers, 256, 384, 384);
    }
    else if (std::strcmp(data_name.c_str(), "S3D") == 0){
        refactor_3D<T>(data_file_prefix, rdata_file_prefix, interp_op, direction_op, layers, 500, 500, 500);
    }
    return;
}

void usage(char* cmd) {
    std::cout << "refactor_data usage: " << cmd <<
                  " data_name data_path -[dataType: f/d]"
                  << std::endl
                  << "example: " << cmd <<
                  " GE ./dataset/GE/ -d" << std::endl;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 0;
    }

    int argv_id = 1;
    std::string data_name = argv[argv_id++];
    std::string data_path = argv[argv_id++];

    int interp_op = 1; // linear:0 cubic:1
    int direction_op = 0; // dimension high -> low
    int layers = 9;

    if((argv[3] + 1)[0] == 'f') {
        layers = 1;
        QoI_compress_preprocess<float>(data_name, data_path, interp_op, direction_op, layers);
    } // precision: 1e-6
    else if((argv[3] + 1)[0] == 'd') {
        layers = 9;
        QoI_compress_preprocess<double>(data_name, data_path, interp_op, direction_op, layers);
    } // precision: 1e-9

    return 0;
}