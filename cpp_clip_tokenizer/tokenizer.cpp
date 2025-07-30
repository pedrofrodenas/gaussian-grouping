#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <array>
#include <algorithm>
#include <cctype>
#include <limits>
#include <curl/curl.h>
#include <zlib.h>

class ReplicatedTokenizer {
private:
    int context_length;
    std::array<wchar_t, 256> byte_encoder;
    std::map<std::wstring, int> encoder;
    std::map<std::pair<std::wstring, std::wstring>, int> bpe_ranks;
    int sot_token_id;
    int eot_token_id;

    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
        size_t realsize = size * nmemb;
        userp->append(static_cast<char*>(contents), realsize);
        return realsize;
    }

    std::string download_bpe(const std::string& url) {
        CURL* curl = curl_easy_init();
        if (!curl) throw std::runtime_error("CURL initialization failed");

        std::string data;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);
        CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) throw std::runtime_error("Failed to download BPE vocab");
        return data;
    }

    std::string decompress_gzip(const std::string& compressed) {
        z_stream zs{};
        zs.zalloc = Z_NULL;
        zs.zfree = Z_NULL;
        zs.opaque = Z_NULL;
        zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(compressed.data()));
        zs.avail_in = compressed.size();

        if (inflateInit2(&zs, 16 + MAX_WBITS) != Z_OK) throw std::runtime_error("zlib init failed");

        std::string decompressed;
        char buffer[16384];
        do {
            zs.next_out = reinterpret_cast<Bytef*>(buffer);
            zs.avail_out = sizeof(buffer);
            int ret = inflate(&zs, Z_NO_FLUSH);
            if (ret == Z_STREAM_ERROR) throw std::runtime_error("zlib decompression error");
            decompressed.append(buffer, sizeof(buffer) - zs.avail_out);
        } while (zs.avail_out == 0);

        inflateEnd(&zs);
        return decompressed;
    }

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
        return result;
    }

    std::vector<int> encode(const std::wstring& text) {
        std::vector<int> bpe_tokens;
        std::wstring cleaned = whitespace_clean(text);
        std::transform(cleaned.begin(), cleaned.end(), cleaned.begin(), towlower);

        // Simple tokenization for sample texts (approximating regex)
        std::vector<std::wstring> tokens;
        std::wstring current;
        for (wchar_t c : cleaned) {
            if (std::isspace(c)) {
                if (!current.empty()) tokens.push_back(current);
                current.clear();
            } else {
                current += c;
            }
        }
        if (!current.empty()) tokens.push_back(current);

        for (const auto& token : tokens) {
            std::string utf8;
            for (wchar_t wc : token) {
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
    ReplicatedTokenizer(int ctx_len = 77) : context_length(ctx_len) {
        std::string bpe_url = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz";
        std::string merges_gz = download_bpe(bpe_url);
        std::string merges_text = decompress_gzip(merges_gz);
        
        std::vector<std::string> merges;
        std::string line;
        for (char c : merges_text) {
            if (c == '\n') {
                if (!line.empty()) merges.push_back(line);
                line.clear();
            } else {
                line += c;
            }
        }
        merges = std::vector<std::string>(merges.begin() + 1, merges.begin() + std::min<size_t>(merges.size(), 49152 - 256 - 2 + 1));

        byte_encoder = bytes_to_unicode();
        std::vector<std::wstring> vocab;
        for (wchar_t c : byte_encoder) vocab.emplace_back(1, c);
        for (wchar_t c : byte_encoder) vocab.push_back(std::wstring(1, c) + L"</w>");
        for (const auto& merge : merges) {
            std::string part1, part2;
            bool first = true;
            for (char c : merge) {
                if (c == ' ') first = false;
                else if (first) part1 += c;
                else part2 += c;
            }
            vocab.push_back(std::wstring(part1.begin(), part1.end()) + std::wstring(part2.begin(), part2.end()));
        }
        std::vector<std::wstring> special_tokens = {L"<start_of_text>", L"<end_of_text>"};
        vocab.insert(vocab.end(), special_tokens.begin(), special_tokens.end());

        for (size_t i = 0; i < vocab.size(); ++i) encoder[vocab[i]] = i;
        for (size_t i = 0; i < merges.size(); ++i) {
            std::string part1, part2;
            bool first = true;
            for (char c : merges[i]) {
                if (c == ' ') first = false;
                else if (first) part1 += c;
                else part2 += c;
            }
            bpe_ranks[{std::wstring(part1.begin(), part1.end()), std::wstring(part2.begin(), part2.end())}] = i;
        }

        sot_token_id = encoder[L"<start_of_text>"];
        eot_token_id = encoder[L"<end_of_text>"];
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
        L"This is a much longer sentence to test truncation and ensure everything works as expected."
    };

    std::cout << "--- Verification ---\n\n2. Tokenizing with ReplicatedTokenizer...\n";
    ReplicatedTokenizer tokenizer;
    auto tokens = tokenizer(sample_texts);

    std::cout << "Replicated tokenizer output shape: [" << tokens.size() << ", " << tokens[0].size() << "]\n";
    std::cout << "Replicated tokens (first example):\n[";
    for (size_t i = 0; i < tokens[0].size(); ++i) {
        std::cout << tokens[0][i];
        if (i < tokens[0].size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    return 0;
}
