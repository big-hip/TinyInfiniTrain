#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
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

TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    CHECK_GT(sequence_length, 0);
    CHECK(std::filesystem::exists(path)) << "Dataset file not found: " << path;

    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open dataset file: " << path;
    const auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    CHECK_EQ(ifs.gcount(), 1024) << "Dataset header is truncated: " << path;

    const uint32_t magic = BytesToType<uint32_t>(header, 0);
    CHECK(kTypeMap.contains(static_cast<int>(magic))) << "Unsupported dataset magic: " << magic;
    const auto type = kTypeMap.at(static_cast<int>(magic));
    const uint32_t version = BytesToType<uint32_t>(header, 4);
    const uint32_t expected_version = type == TinyShakespeareType::kUINT16 ? 1 : 7;
    CHECK_EQ(version, expected_version) << "Unsupported dataset version: " << version;
    const uint32_t num_tokens = BytesToType<uint32_t>(header, 8);
    CHECK_GE(num_tokens, sequence_length + 1);

    const size_t token_size = kTypeToSize.at(type);
    const auto file_size = std::filesystem::file_size(path);
    const size_t data_size = static_cast<size_t>(num_tokens) * token_size;
    CHECK_GE(file_size, 1024 + data_size) << "Dataset token data is truncated: " << path;

    std::vector<uint8_t> raw_data(data_size);
    ifs.read(reinterpret_cast<char *>(raw_data.data()), static_cast<std::streamsize>(raw_data.size()));
    CHECK_EQ(ifs.gcount(), static_cast<std::streamsize>(raw_data.size())) << "Dataset token data is truncated: " << path;

    infini_train::Tensor tensor({static_cast<int64_t>(num_tokens)}, DataType::kINT64);
    auto *tensor_ptr = static_cast<int64_t *>(tensor.DataPtr());
    if (type == TinyShakespeareType::kUINT16) {
        for (uint32_t idx = 0; idx < num_tokens; ++idx) {
            tensor_ptr[idx] = BytesToType<uint16_t>(raw_data, static_cast<size_t>(idx) * token_size);
        }
    } else {
        for (uint32_t idx = 0; idx < num_tokens; ++idx) {
            tensor_ptr[idx] = BytesToType<uint32_t>(raw_data, static_cast<size_t>(idx) * token_size);
        }
    }

    const int64_t num_chunks =
        (static_cast<int64_t>(num_tokens) + static_cast<int64_t>(sequence_length) - 1) / sequence_length;
    return {type, {num_chunks, static_cast<int64_t>(sequence_length)}, std::move(tensor)};
}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)), sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)), num_samples_(text_file_.dims[0] - 1) {
    CHECK_GE(text_file_.dims.size(), 2);
    CHECK_GT(text_file_.dims[0], 0);
}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
