#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
namespace {
void CheckMatmulInputs(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    CHECK(input->GetDevice().IsCPU());
    CHECK(other->GetDevice().IsCPU());
    CHECK(input->GetDevice() == other->GetDevice());
    CHECK(input->Dtype() == DataType::kFLOAT32);
    CHECK(other->Dtype() == DataType::kFLOAT32);
    CHECK_EQ(input->Dims().size(), other->Dims().size());
    CHECK_GE(input->Dims().size(), 2);
    for (size_t i = 0; i + 2 < input->Dims().size(); ++i) {
        CHECK_EQ(input->Dims()[i], other->Dims()[i]);
    }
    CHECK_EQ(input->Dims().back(), other->Dims()[other->Dims().size() - 2]);
}

int64_t BatchCount(const std::vector<int64_t> &dims) {
    return std::accumulate(dims.begin(), dims.end() - 2, int64_t{1}, std::multiplies<int64_t>{});
}
} // namespace

std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    CheckMatmulInputs(input, other);

    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const int64_t batch_count = BatchCount(input_dims);
    const int64_t rows = input_dims[input_dims.size() - 2];
    const int64_t inner = input_dims.back();
    const int64_t cols = other_dims.back();

    auto output_dims = input_dims;
    output_dims.back() = cols;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());

    const auto *input_ptr = static_cast<const float *>(input->DataPtr());
    const auto *other_ptr = static_cast<const float *>(other->DataPtr());
    auto *output_ptr = static_cast<float *>(output->DataPtr());
    for (int64_t batch = 0; batch < batch_count; ++batch) {
        const auto *input_batch = input_ptr + batch * rows * inner;
        const auto *other_batch = other_ptr + batch * inner * cols;
        auto *output_batch = output_ptr + batch * rows * cols;
        for (int64_t row = 0; row < rows; ++row) {
            for (int64_t col = 0; col < cols; ++col) {
                float value = 0.0f;
                for (int64_t k = 0; k < inner; ++k) {
                    value += input_batch[row * inner + k] * other_batch[k * cols + col];
                }
                output_batch[row * cols + col] = value;
            }
        }
    }
    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    CheckMatmulInputs(input, other);

    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    auto output_dims = input_dims;
    output_dims.back() = other_dims.back();
    CHECK(grad_output->GetDevice().IsCPU());
    CHECK(grad_output->Dtype() == DataType::kFLOAT32);
    CHECK(grad_output->Dims() == output_dims);

    const int64_t batch_count = BatchCount(input_dims);
    const int64_t rows = input_dims[input_dims.size() - 2];
    const int64_t inner = input_dims.back();
    const int64_t cols = other_dims.back();
    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, input->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32, other->GetDevice());

    const auto *input_ptr = static_cast<const float *>(input->DataPtr());
    const auto *other_ptr = static_cast<const float *>(other->DataPtr());
    const auto *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    auto *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());
    auto *grad_other_ptr = static_cast<float *>(grad_other->DataPtr());
    for (int64_t batch = 0; batch < batch_count; ++batch) {
        const auto *input_batch = input_ptr + batch * rows * inner;
        const auto *other_batch = other_ptr + batch * inner * cols;
        const auto *grad_output_batch = grad_output_ptr + batch * rows * cols;
        auto *grad_input_batch = grad_input_ptr + batch * rows * inner;
        auto *grad_other_batch = grad_other_ptr + batch * inner * cols;
        for (int64_t row = 0; row < rows; ++row) {
            for (int64_t k = 0; k < inner; ++k) {
                float value = 0.0f;
                for (int64_t col = 0; col < cols; ++col) {
                    value += grad_output_batch[row * cols + col] * other_batch[k * cols + col];
                }
                grad_input_batch[row * inner + k] = value;
            }
        }
        for (int64_t k = 0; k < inner; ++k) {
            for (int64_t col = 0; col < cols; ++col) {
                float value = 0.0f;
                for (int64_t row = 0; row < rows; ++row) {
                    value += input_batch[row * inner + k] * grad_output_batch[row * cols + col];
                }
                grad_other_batch[k * cols + col] = value;
            }
        }
    }
    return {grad_input, grad_other};
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {
    /*
    transpose:  output = input * weight^T + bias
    output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]

    !transpose: output = input * weight + bias
    output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    const int out_features = weight_dims[transpose ? 0 : 1];

    if (bias) {
        const auto &bias_dims = bias->Dims();
        CHECK_EQ(bias_dims.size(), 1);
        CHECK_EQ(bias_dims[0], out_features);
    }

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    if (transpose) {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix().transpose();
    } else {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix();
    }

    if (bias) {
        output->EigenMatrix().rowwise() += bias->EigenVector();
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    /*
    transpose: grad_input = grad_output * weight
    grad_input[*, in_features] = grad_output[*, out_features] * weight[out_features, in_features]
    grad_weight[out_features, in_features] = grad_output[*, out_features]^T * input[*, in_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)

    !transpose: grad_input = grad_output * weight^T
    grad_input[*, in_features] = grad_output[_, out_features] * weight[in_features, out_features]^T
    grad_weight[in_features, out_features] = input[*, in_features]^T * grad_output[*, out_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32);
    }

    if (transpose) {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix();
        grad_weight->EigenMatrix() = grad_output->EigenMatrix().transpose() * input->EigenMatrix();
    } else {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix().transpose();
        grad_weight->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
    }
    if (bias) {
        grad_bias->EigenVector() = grad_output->EigenMatrix().colwise().sum();
    }

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_LINEAR_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_LINEAR_KERNEL(MatmulForward)
REGISTER_CPU_LINEAR_KERNEL(MatmulBackward)
REGISTER_CPU_LINEAR_KERNEL(LinearForward)
REGISTER_CPU_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CPU_LINEAR_KERNEL
