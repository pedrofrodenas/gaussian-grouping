#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <filesystem>
#include <regex>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include "cnpy.h"

namespace fs = std::filesystem;

// Float16 to Float32 conversion functions
class Float16Converter {
public:
    // Convert IEEE 754 half-precision to single-precision
    static float half_to_float(uint16_t h) {
        uint32_t sign = (h & 0x8000) << 16;
        int32_t exp = (h & 0x7C00) >> 10;
        uint32_t mantissa = h & 0x03FF;
        
        if (exp == 0) {
            if (mantissa == 0) {
                // Zero
                return *reinterpret_cast<float*>(&sign);
            } else {
                // Denormalized number
                exp = -14;
                while ((mantissa & 0x400) == 0) {
                    mantissa <<= 1;
                    exp--;
                }
                mantissa &= 0x3FF;
                exp += 127 - 15 + 1;
            }
        } else if (exp == 0x1F) {
            // Infinity or NaN
            exp = 0xFF;
        } else {
            // Normalized number
            exp += 127 - 15;
        }
        
        uint32_t result = sign | (exp << 23) | (mantissa << 13);
        return *reinterpret_cast<float*>(&result);
    }
    
    // Convert array of float16 to float32
    static std::vector<float> convert_array(const uint16_t* data, size_t count) {
        std::vector<float> result(count);
        for (size_t i = 0; i < count; ++i) {
            result[i] = half_to_float(data[i]);
        }
        return result;
    }
};

class NumpyArrayLoader {
private:
    std::map<int, cnpy::NpyArray> arrays_;
    std::map<int, std::vector<float>> converted_arrays_; // Store converted float32 data
    std::string folder_path_;

public:
    explicit NumpyArrayLoader(const std::string& folder_path) 
        : folder_path_(folder_path) {}

    // Load all array_*.npy files from the specified folder
    bool loadArrays() {
        try {
            // Check if folder exists
            if (!fs::exists(folder_path_) || !fs::is_directory(folder_path_)) {
                std::cerr << "Error: Folder " << folder_path_ << " does not exist or is not a directory" << std::endl;
                return false;
            }

            // Regular expression to match array_X.npy pattern
            std::regex pattern(R"(array_(\d+)\.npy)");
            std::smatch matches;

            // Iterate through all files in the directory
            for (const auto& entry : fs::directory_iterator(folder_path_)) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().filename().string();
                    
                    // Check if filename matches the pattern
                    if (std::regex_match(filename, matches, pattern)) {
                        try {
                            // Extract the number from filename
                            int array_id = std::stoi(matches[1].str());
                            
                            // Load the .npy file
                            std::string full_path = entry.path().string();
                            cnpy::NpyArray array = cnpy::npy_load(full_path);
                            
                            // Check if this is float16 data and convert to float32
                            if (array.word_size == 2) { // float16 is 2 bytes
                                const uint16_t* data = array.data<uint16_t>();
                                converted_arrays_[array_id] = Float16Converter::convert_array(data, array.num_vals);
                                
                                std::cout << "Loaded and converted " << filename << " (float16->float32) with ID " << array_id;
                            } else {
                                std::cout << "Loaded " << filename << " with ID " << array_id;
                            }
                            
                            // Store original array
                            arrays_[array_id] = std::move(array);
                            
                            std::cout << " (shape: ";
                            for (size_t i = 0; i < arrays_[array_id].shape.size(); ++i) {
                                std::cout << arrays_[array_id].shape[i];
                                if (i < arrays_[array_id].shape.size() - 1) std::cout << "x";
                            }
                            std::cout << ", word_size: " << arrays_[array_id].word_size << " bytes)" << std::endl;
                            
                        } catch (const std::exception& e) {
                            std::cerr << "Error loading " << filename << ": " << e.what() << std::endl;
                        }
                    }
                }
            }

            std::cout << "Successfully loaded " << arrays_.size() << " arrays" << std::endl;
            return !arrays_.empty();

        } catch (const std::exception& e) {
            std::cerr << "Error accessing directory: " << e.what() << std::endl;
            return false;
        }
    }

    // Get array by ID (returns original data)
    const cnpy::NpyArray* getArray(int id) const {
        auto it = arrays_.find(id);
        return (it != arrays_.end()) ? &it->second : nullptr;
    }

    // Get converted float32 data by ID (returns nullptr if not float16 originally)
    const std::vector<float>* getFloat32Array(int id) const {
        auto it = converted_arrays_.find(id);
        return (it != converted_arrays_.end()) ? &it->second : nullptr;
    }

    // Check if array was originally float16
    bool isFloat16Array(int id) const {
        const cnpy::NpyArray* array = getArray(id);
        return array && array->word_size == 2;
    }

    // Get float32 data (either converted from float16 or original float32)
    const float* getFloatData(int id) const {
        if (isFloat16Array(id)) {
            const std::vector<float>* converted = getFloat32Array(id);
            return converted ? converted->data() : nullptr;
        } else {
            const cnpy::NpyArray* array = getArray(id);
            if (array && array->word_size == sizeof(float)) {
                return array->data<float>();
            }
        }
        return nullptr;
    }

    // Get all array IDs
    std::vector<int> getArrayIds() const {
        std::vector<int> ids;
        for (const auto& pair : arrays_) {
            ids.push_back(pair.first);
        }
        std::sort(ids.begin(), ids.end());
        return ids;
    }

    // Get number of loaded arrays
    size_t size() const {
        return arrays_.size();
    }

    // Print array information
    void printArrayInfo(int id) const {
        const cnpy::NpyArray* array = getArray(id);
        if (!array) {
            std::cout << "Array " << id << " not found" << std::endl;
            return;
        }

        std::cout << "Array " << id << ":" << std::endl;
        std::cout << "  Shape: ";
        for (size_t i = 0; i < array->shape.size(); ++i) {
            std::cout << array->shape[i];
            if (i < array->shape.size() - 1) std::cout << " x ";
        }
        std::cout << std::endl;
        std::cout << "  Original data type size: " << array->word_size << " bytes";
        if (array->word_size == 2) {
            std::cout << " (float16, converted to float32)";
        }
        std::cout << std::endl;
        std::cout << "  Total elements: " << array->num_vals << std::endl;
        
        // Print first few elements using float32 data
        const float* data = getFloatData(id);
        if (data) {
            std::cout << "  First 10 elements (float32): ";
            size_t print_count = std::min(static_cast<size_t>(10), array->num_vals);
            for (size_t i = 0; i < print_count; ++i) {
                std::cout << data[i];
                if (i < print_count - 1) std::cout << ", ";
            }
            if (array->num_vals > 10) std::cout << " ...";
            std::cout << std::endl;
        }
    }

    // Print information for all loaded arrays
    void printAllArraysInfo() const {
        std::vector<int> ids = getArrayIds();
        for (int id : ids) {
            printArrayInfo(id);
            std::cout << std::endl;
        }
    }
};

// Example usage
int main() {
    // Change this to your actual folder path
    std::string folder_path = "/home/prodenas/Projects/gaussian-grouping/output/figuritas/point_cloud_object_removal/iteration_30000";
    
    NumpyArrayLoader loader(folder_path);
    
    // Load all arrays
    if (!loader.loadArrays()) {
        std::cerr << "Failed to load arrays from " << folder_path << std::endl;
        return 1;
    }

    // Print information about all loaded arrays
    std::cout << "\n=== Array Information ===" << std::endl;
    loader.printAllArraysInfo();

    // Example: Access specific arrays
    std::cout << "=== Accessing Specific Arrays ===" << std::endl;
    std::vector<int> ids = loader.getArrayIds();
    
    for (int id : ids) {
        const cnpy::NpyArray* array = loader.getArray(id);
        if (array) {
            const float* data = loader.getFloatData(id); // This handles both float16->float32 and native float32
            
            if (data) {
                std::cout << "Array " << id << " - Shape: ";
                for (size_t i = 0; i < array->shape.size(); ++i) {
                    std::cout << array->shape[i];
                    if (i < array->shape.size() - 1) std::cout << "x";
                }
                
                // Example: access element at position [0,0] (assuming 2D array)
                if (array->shape.size() >= 2) {
                    std::cout << " - Element [0,0] (as float32): " << data[0] << std::endl;
                }
                
                if (loader.isFloat16Array(id)) {
                    std::cout << "  (Originally float16, converted to float32)" << std::endl;
                }
            } else {
                std::cout << "Array " << id << " - Unsupported data type (word_size: " 
                          << array->word_size << " bytes)" << std::endl;
            }
        }
    }

    return 0;
}