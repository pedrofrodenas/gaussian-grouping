#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <array>
#include <algorithm>
#include <cctype>
#include <limits>
#include <fstream>
#include <regex>
#include <codecvt>
#include <locale>

class ReplicatedTokenizer {
private:
    int context_length;
    std::array<wchar_t, 256> byte_encoder;
    std::map<wchar_t, unsigned char> byte_decoder;
    std::map<std::wstring, int> encoder;
    std::map<int, std::wstring> decoder;  // Fixed: int -> std::wstring
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
        // Simple HTML unescaping for common entities
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
        while (std::getline(file, line) && merge_count < 48894) { // Changed from 48895 to 48894
            if (!line.empty()) merges.push_back(line);
            merge_count++;
        }
        file.close();

        byte_encoder = bytes_to_unicode();
        for (size_t i = 0; i < 256; ++i) {
            byte_decoder[byte_encoder[i]] = static_cast<unsigned char>(i);
        }

        // Define bs as in bytes_to_unicode
        std::vector<int> bs;
        for (int i = L'!'; i <= L'~'; ++i) bs.push_back(i);     // 33-126
        for (int i = L'¡'; i <= L'¬'; ++i) bs.push_back(i);     // 161-172
        for (int i = L'®'; i <= L'ÿ'; ++i) bs.push_back(i);     // 174-255
        std::set<int> bs_set(bs.begin(), bs.end());
        for (int b = 0; b < 256; ++b) {
            if (bs_set.find(b) == bs_set.end()) {
                bs.push_back(b);                                // 0-32, 127, 173
            }
        }

        // Build vocab using bs order
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
};

int main() {
    std::vector<std::wstring> sample_texts = {
        L"a photo of a cat",
        L"a drawing of a dog on a skateboard",
        L"This is a much longer sentence to test truncation and ensure everything works as expected.",
        L"don't stop"
    };
    std::cout << "--- Verification ---\n\n2. Tokenizing with ReplicatedTokenizer...\n";
    try {
        ReplicatedTokenizer tokenizer("/home/prodenas/Projects/gaussian-grouping/cpp_clip_tokenizer/bpe_simple_vocab_16e6.txt");
        auto tokens = tokenizer(sample_texts);
        std::cout << "Replicated tokenizer output shape: [" << tokens.size() << ", " << tokens[0].size() << "]\n";
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << "Replicated tokens (example " << i + 1 << "):\n[";
            for (size_t j = 0; j < tokens[i].size(); ++j) {
                std::cout << tokens[i][j];
                if (j < tokens[i].size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    return 0;
}