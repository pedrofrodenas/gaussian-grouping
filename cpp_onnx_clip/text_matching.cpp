#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <regex>
#include <codecvt>
#include <locale>
#include <algorithm>
#include <cctype>
#include <limits>
#include <fstream>
#include <filesystem>
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

// ReplicatedTokenizer class definition (unchanged except for adding get_eot_token_id)
class ReplicatedTokenizer {
private:
    int context_length;
    std::array<wchar_t, 256> byte_encoder;
    std::map<wchar_t, unsigned char> byte_decoder;
    std::map<std::wstring, int> encoder;
    std::map<int, std::wstring> decoder;
    std::map<std::pair<std::wstring, std::wstring>, int> bpe_ranks;
    std::map<std::wstring, std::wstring> cache;
    std::wregex pat;
    int sot_token_id;
    int eot_token_id;

    std::array<wchar_t, 256> bytes_to_unicode() {
        std::array<wchar_t, 256> mapping{};
        std::vector<int> bs;
        for (int i = L'!'; i <= L'~'; ++i) bs.push_back(i);
        for (int i = L'¡'; i <= L'¬'; ++i) bs.push_back(i);
        for (int i = L'®'; i <= L'ÿ'; ++i) bs.push_back(i);
        std::set<int> bs_set(bs.begin(), bs.end());
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (bs_set.count(b)) mapping[b] = static_cast<wchar_t>(b);
            else mapping[b] = static_cast<wchar_t>(256 + n++);
        }
        return mapping;
    }

    std::set<std::pair<std::wstring, std::wstring>> get_pairs(const std::vector<std::wstring>& word) {
        std::set<std::pair<std::wstring, std::wstring>> pairs;
        if (word.size() < 2) return pairs;
        for (size_t i = 0; i < word.size() - 1; ++i) {
            pairs.emplace(word[i], word[i + 1]);
        }
        return pairs;
    }

    std::wstring basic_clean(const std::wstring& text) {
        std::wstring result;
        std::map<std::wstring, wchar_t> entities = {
            {L"&amp;", L'&'}, {L"&lt;", L'<'}, {L"&gt;", L'>'},
            {L"&quot;", L'"'}, {L"&apos;", L'\''}, {L"&nbsp;", L' '}
        };
        size_t i = 0;
        while (i < text.length()) {
            bool matched = false;
            for (const auto& [entity, ch] : entities) {
                if (text.substr(i, entity.length()) == entity) {
                    result += ch;
                    i += entity.length();
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                result += text[i];
                i++;
            }
        }
        return result;
    }

    std::wstring whitespace_clean(const std::wstring& text) {
        std::wstring cleaned, word;
        for (wchar_t c : text) {
            if (std::isspace(c)) {
                if (!word.empty()) {
                    if (!cleaned.empty()) cleaned += L" ";
                    cleaned += word;
                    word.clear();
                }
            } else {
                word += c;
            }
        }
        if (!word.empty()) {
            if (!cleaned.empty()) cleaned += L" ";
            cleaned += word;
        }
        return cleaned;
    }

    std::wstring bpe(const std::wstring& token) {
        if (cache.count(token)) return cache[token];
        if (token.empty()) return token + L"</w>";
        std::vector<std::wstring> word;
        for (size_t i = 0; i < token.size() - 1; ++i) word.emplace_back(1, token[i]);
        word.push_back(std::wstring(1, token.back()) + L"</w>");
        while (true) {
            auto pairs = get_pairs(word);
            if (pairs.empty()) break;
            auto min_pair = *std::min_element(pairs.begin(), pairs.end(),
                [this](const auto& p1, const auto& p2) {
                    int r1 = bpe_ranks.count(p1) ? bpe_ranks.at(p1) : std::numeric_limits<int>::max();
                    int r2 = bpe_ranks.count(p2) ? bpe_ranks.at(p2) : std::numeric_limits<int>::max();
                    return r1 < r2;
                });
            if (!bpe_ranks.count(min_pair)) break;
            std::vector<std::wstring> new_word;
            size_t i = 0;
            while (i < word.size()) {
                if (i < word.size() - 1 && word[i] == min_pair.first && word[i + 1] == min_pair.second) {
                    new_word.push_back(min_pair.first + min_pair.second);
                    i += 2;
                } else {
                    new_word.push_back(word[i]);
                    i += 1;
                }
            }
            word = std::move(new_word);
        }
        std::wstring result;
        for (size_t i = 0; i < word.size(); ++i) {
            result += word[i];
            if (i < word.size() - 1) result += L" ";
        }
        cache[token] = result;
        return result;
    }

    std::vector<int> encode(const std::wstring& text) {
        std::vector<int> bpe_tokens;
        std::wstring cleaned = whitespace_clean(basic_clean(text));
        std::transform(cleaned.begin(), cleaned.end(), cleaned.begin(), towlower);

        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        std::string utf8_text = converter.to_bytes(cleaned);

        std::regex pat_regex("<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]+|[^\\s a-z0-9]+",
                             std::regex::ECMAScript | std::regex::icase);
        std::sregex_iterator it(utf8_text.begin(), utf8_text.end(), pat_regex);
        std::sregex_iterator end;

        for (; it != end; ++it) {
            std::string token = it->str();
            std::wstring wtoken = converter.from_bytes(token);
            std::string utf8;
            for (wchar_t wc : wtoken) {
                if (wc <= 0x7F) utf8 += static_cast<char>(wc);
                else if (wc <= 0x7FF) {
                    utf8 += static_cast<char>(0xC0 | (wc >> 6));
                    utf8 += static_cast<char>(0x80 | (wc & 0x3F));
                } else {
                    utf8 += static_cast<char>(0xE0 | (wc >> 12));
                    utf8 += static_cast<char>(0x80 | ((wc >> 6) & 0x3F));
                    utf8 += static_cast<char>(0x80 | (wc & 0x3F));
                }
            }
            std::wstring mapped;
            for (char c : utf8) mapped += byte_encoder[static_cast<unsigned char>(c)];
            std::wstring bpe_result = bpe(mapped);
            std::wstring current_subword;
            for (wchar_t c : bpe_result) {
                if (c == L' ') {
                    if (!current_subword.empty()) {
                        bpe_tokens.push_back(encoder.at(current_subword));
                        current_subword.clear();
                    }
                } else {
                    current_subword += c;
                }
            }
            if (!current_subword.empty()) bpe_tokens.push_back(encoder.at(current_subword));
        }
        return bpe_tokens;
    }

public:
    ReplicatedTokenizer(const std::string& merges_file_path, int ctx_len = 77) : context_length(ctx_len) {
        std::ifstream file(merges_file_path);
        if (!file.is_open()) throw std::runtime_error("Failed to open merges file");
        std::string line;
        std::vector<std::string> merges;
        std::getline(file, line); // Skip version line
        size_t merge_count = 0;
        while (std::getline(file, line) && merge_count < 48894) {
            if (!line.empty()) merges.push_back(line);
            merge_count++;
        }
        file.close();

        byte_encoder = bytes_to_unicode();
        for (size_t i = 0; i < 256; ++i) {
            byte_decoder[byte_encoder[i]] = static_cast<unsigned char>(i);
        }

        std::vector<int> bs;
        for (int i = L'!'; i <= L'~'; ++i) bs.push_back(i);
        for (int i = L'¡'; i <= L'¬'; ++i) bs.push_back(i);
        for (int i = L'®'; i <= L'ÿ'; ++i) bs.push_back(i);
        std::set<int> bs_set(bs.begin(), bs.end());
        for (int b = 0; b < 256; ++b) {
            if (bs_set.find(b) == bs_set.end()) {
                bs.push_back(b);
            }
        }

        std::vector<std::wstring> vocab;
        for (int b : bs) {
            vocab.push_back(std::wstring(1, byte_encoder[b]));
        }
        for (int b : bs) {
            vocab.push_back(std::wstring(1, byte_encoder[b]) + L"</w>");
        }

        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        for (const auto& merge : merges) {
            std::string part1, part2;
            bool first = true;
            for (char c : merge) {
                if (c == ' ') first = false;
                else if (first) part1 += c;
                else part2 += c;
            }
            vocab.push_back(converter.from_bytes(part1) + converter.from_bytes(part2));
        }
        std::vector<std::wstring> special_tokens = {L"<start_of_text>", L"<end_of_text>"};
        vocab.insert(vocab.end(), special_tokens.begin(), special_tokens.end());

        for (size_t i = 0; i < vocab.size(); ++i) {
            encoder[vocab[i]] = static_cast<int>(i);
            decoder[static_cast<int>(i)] = vocab[i];
        }
        for (size_t i = 0; i < merges.size(); ++i) {
            std::string part1, part2;
            bool first = true;
            for (char c : merges[i]) {
                if (c == ' ') first = false;
                else if (first) part1 += c;
                else part2 += c;
            }
            bpe_ranks[{converter.from_bytes(part1), converter.from_bytes(part2)}] = static_cast<int>(i);
        }
        for (const auto& t : special_tokens) cache[t] = t;
        sot_token_id = encoder[L"<start_of_text>"];
        eot_token_id = encoder[L"<end_of_text>"];
        pat = std::wregex(L"<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
                          std::regex::ECMAScript | std::regex::icase);
    }

    std::vector<std::vector<int>> operator()(const std::vector<std::wstring>& texts, int ctx_len = -1) {
        int effective_ctx = (ctx_len == -1) ? context_length : ctx_len;
        std::vector<std::vector<int>> all_tokens;
        for (const auto& text : texts) {
            std::vector<int> encoded = {sot_token_id};
            auto token_ids = encode(text);
            encoded.insert(encoded.end(), token_ids.begin(), token_ids.end());
            encoded.push_back(eot_token_id);
            if (encoded.size() > static_cast<size_t>(effective_ctx)) {
                encoded.resize(effective_ctx);
                encoded.back() = eot_token_id;
            } else {
                encoded.resize(effective_ctx, 0);
            }
            all_tokens.push_back(encoded);
        }
        return all_tokens;
    }

    // Added getter for eot_token_id
    int get_eot_token_id() const { return eot_token_id; }
};

int main() {
    try {
        // 1. Instantiate the tokenizer
        ReplicatedTokenizer tokenizer("/home/prodenas/Projects/gaussian-grouping/cpp_clip_tokenizer/bpe_simple_vocab_16e6.txt");
        std::vector<std::wstring> texts = {L"a photo of a cat", L"another text"};
        auto tokens = tokenizer(texts);

        // 2. Prepare eot_indices
        int eot_id = tokenizer.get_eot_token_id();
        std::vector<int64_t> eot_indices;
        for (const auto& seq : tokens) {
            auto it = std::find(seq.begin(), seq.end(), eot_id);
            if (it != seq.end()) {
                int index = std::distance(seq.begin(), it);
                eot_indices.push_back(static_cast<int64_t>(index));
            } else {
                // EOT should always be present, but handle the edge case
                throw std::runtime_error("EOT token not found in tokenized sequence");
            }
        }

        // 3. Prepare flat text_tokens
        std::vector<int64_t> flat_tokens;
        for (const auto& seq : tokens) {
            for (int id : seq) {
                flat_tokens.push_back(static_cast<int64_t>(id));
            }
        }

        size_t batch_size = tokens.size();
        size_t context_length = tokens[0].size(); // 77 by default

        // 4. Load the ONNX model
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "OpenCLIP_Inference");
        Ort::SessionOptions session_options;
        Ort::Session session(env, "/home/prodenas/Projects/gaussian-grouping/cpp_onnx_clip/text_encoder.onnx", session_options);

        // 5. Create input tensors
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<int64_t> text_tokens_shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(context_length)};
        Ort::Value text_tokens_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, flat_tokens.data(), flat_tokens.size(), text_tokens_shape.data(), text_tokens_shape.size()
        );

        std::vector<int64_t> eot_indices_shape = {static_cast<int64_t>(batch_size)};
        Ort::Value eot_indices_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, eot_indices.data(), eot_indices.size(), eot_indices_shape.data(), eot_indices_shape.size()
        );

        // 6. Define input and output names
        const char* input_names[] = {"text_tokens", "eot_indices"};
        const char* output_names[] = {"text_features"};

        // 7. Run inference
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(text_tokens_tensor));
        inputs.push_back(std::move(eot_indices_tensor));
        auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(), output_names, 1);

        // 8. Process the output
        Ort::Value& output_tensor = outputs[0];
        float* output_data = output_tensor.GetTensorMutableData<float>();
        auto output_shape_info = output_tensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_shape = output_shape_info.GetShape();
        size_t feature_dim = output_shape[1]; // Typically 512 for ViT-B-32

        std::cout << "Output shape: [" << output_shape[0] << ", " << output_shape[1] << "]\n";
        for (size_t i = 0; i < batch_size; ++i) {
            std::cout << "Text " << i + 1 << " features: ";
            for (size_t j = 0; j < std::min<size_t>(feature_dim, 5); ++j) { // Print first 5 features for brevity
                std::cout << output_data[i * feature_dim + j] << " ";
            }
            std::cout << "...\n";
        }

        std::string folder_path = "/home/prodenas/Projects/gaussian-grouping/output/figuritas/point_cloud_object_removal/iteration_30000";

        NumpyArrayLoader loader(folder_path);

        // Load all arrays
        if (!loader.loadArrays()) {
            std::cerr << "Failed to load arrays from " << folder_path << std::endl;
            return 1;
        }


    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << " (Code: " << e.GetOrtErrorCode() << ")\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
