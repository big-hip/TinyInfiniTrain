#include "example/common/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

#ifdef USE_CUDA
#include "cuda_runtime_api.h"
#endif

namespace infini_train {

constexpr uint32_t kGpt2Eot = 50256;
constexpr uint32_t kLLaMA3Eot = 128001;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr uint64_t kRngState = 1337;

using Version = Tokenizer::Version;

const std::unordered_map<uint32_t, uint32_t> kEotMap = {
    {20240328, kGpt2Eot},   // GPT-2
    {20240801, kLLaMA3Eot}, // LLaMA-3
};

const std::unordered_map<uint32_t, std::vector<uint32_t>> kPromptMap = {
    // e.g. "The meaning of life is"
    // ref: https://tiktokenizer.vercel.app/
    {20240328, std::vector<uint32_t>{464, 3616, 286, 1204, 318}}, // GPT-2
    {20240801, std::vector<uint32_t>{791, 7438, 315, 2324, 374}}, // LLaMA-3
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

unsigned int RandomU32(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * kRandomU32Multiplier) >> 32;
}

float RandomF32(uint64_t &state) { // random float32 in [0,1)
    return (RandomU32(state) >> 8) / kF32Divisor;
}

int SampleMult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from RandomF32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

Tokenizer::Tokenizer(const std::string &filepath) {
    CHECK(std::filesystem::exists(filepath)) << "Tokenizer file not found: " << filepath;
    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open tokenizer file: " << filepath;

    const auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    CHECK_EQ(ifs.gcount(), 1024) << "Tokenizer header is truncated: " << filepath;
    magic_number_ = BytesToType<uint32_t>(header, 0);
    CHECK(kEotMap.contains(magic_number_)) << "Unsupported tokenizer magic: " << magic_number_;
    const auto version = BytesToType<uint32_t>(header, 4);
    CHECK(version == static_cast<uint32_t>(Version::kV1) || version == static_cast<uint32_t>(Version::kV2))
        << "Unsupported tokenizer version: " << version;
    vocab_size_ = BytesToType<uint32_t>(header, 8);
    CHECK_GT(vocab_size_, 0);

    token_table_.resize(vocab_size_);
    for (uint32_t token_id = 0; token_id < vocab_size_; ++token_id) {
        uint8_t token_length = 0;
        ifs.read(reinterpret_cast<char *>(&token_length), sizeof(token_length));
        CHECK_EQ(ifs.gcount(), static_cast<std::streamsize>(sizeof(token_length)))
            << "Tokenizer vocabulary is truncated at token " << token_id;

        std::string token(token_length, '\0');
        ifs.read(token.data(), static_cast<std::streamsize>(token_length));
        CHECK_EQ(ifs.gcount(), static_cast<std::streamsize>(token_length))
            << "Tokenizer vocabulary is truncated at token " << token_id;
        token_table_[token_id] = std::move(token);
    }
    eot_token_ = kEotMap.at(magic_number_);
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    if (token_id >= token_table_.size()) {
        LOG(ERROR) << "Token id out of range: " << token_id;
        return {};
    }
    return token_table_[token_id];
}

void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    CHECK_GT(batch_size, 0);
    CHECK_GT(sequence_length, 0);
    std::vector<int64_t> dims;
    dims.assign({batch_size, sequence_length});
    // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(4, 64)
    infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
    int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
    for (int i = 0; i < batch_size * sequence_length; ++i) { x_buff[i] = eot_token_; }

    // Give some contexts: "The meaning of life is "
    auto prompt = kPromptMap.at(magic_number_);
    auto prompt_len = prompt.size();
    for (int i = 0; i < prompt_len; ++i) { x_buff[i] = prompt[i]; }
    std::cout << "The meaning of life is";

    CHECK_LE(prompt_len, sequence_length);
    uint64_t rng_state = kRngState;
    LOG(INFO) << "start generate text:";
    for (uint32_t t = prompt_len; t < text_length; ++t) {
        auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
        const auto outputs = model.Forward({x});
        CHECK(!outputs.empty());
        const auto &logits = outputs[0];
        CHECK(logits != nullptr);
        CHECK_EQ(logits->Dims().size(), 3);
        CHECK_EQ(logits->Dims()[0], batch_size);
        CHECK_EQ(logits->Dims()[1], sequence_length);
        CHECK_EQ(logits->Dims()[2], vocab_size_);

        auto logits_cpu = logits->To(Device());
#ifdef USE_CUDA
        if (logits->GetDevice().IsCUDA()) {
            CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
        }
#endif
        const auto *logits_ptr = static_cast<const float *>(logits_cpu.DataPtr());
        const uint32_t position = std::min(t, sequence_length - 1);
        const auto *last_logits = logits_ptr + position * vocab_size_;

        std::vector<float> probabilities(vocab_size_);
        const float max_logit = *std::max_element(last_logits, last_logits + vocab_size_);
        float probability_sum = 0.0f;
        if (std::isfinite(max_logit)) {
            for (uint32_t token_id = 0; token_id < vocab_size_; ++token_id) {
                probabilities[token_id] = std::exp(last_logits[token_id] - max_logit);
                probability_sum += probabilities[token_id];
            }
        }
        if (!(probability_sum > 0.0f) || !std::isfinite(probability_sum)) {
            const float uniform_probability = 1.0f / static_cast<float>(vocab_size_);
            std::fill(probabilities.begin(), probabilities.end(), uniform_probability);
        } else {
            for (auto &probability : probabilities) { probability /= probability_sum; }
        }

        const uint32_t next_token = static_cast<uint32_t>(SampleMult(probabilities.data(), vocab_size_, RandomF32(rng_state)));
        if (t < sequence_length) {
            x_buff[t] = next_token;
        } else {
            std::memmove(x_buff, x_buff + 1, (sequence_length - 1) * sizeof(int64_t));
            x_buff[sequence_length - 1] = next_token;
        }
        std::cout << Decode(next_token) << std::flush;
    }
    std::cout << std::endl;
}
} // namespace infini_train
