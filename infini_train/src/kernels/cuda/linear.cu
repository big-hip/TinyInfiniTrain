#include "cublas_v2.h"
#include "glog/logging.h"
#include <cub/block/block_reduce.cuh>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess) {                                                                                   \
            LOG(FATAL) << "CUDA Error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;       \
        }                                                                                                              \
    } while (0)

#define CUBLAS_CHECK(call)                                                                                             \
    do {                                                                                                               \
        cublasStatus_t status = call;                                                                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                         \
            LOG(FATAL) << "CUBLAS Error: " << cublasGetStatusString(status) << " at " << __FILE__ << ":" << __LINE__;  \
        }                                                                                                              \
    } while (0)

namespace {
void CheckMatmulInputs(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    CHECK(input->GetDevice().IsCUDA());
    CHECK(other->GetDevice().IsCUDA());
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

__global__ void MatmulForwardKernel(const float *input, const float *other, float *output, int64_t batch_count,
                                    int64_t rows, int64_t inner, int64_t cols) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch_count * rows * cols;
    if (index >= total) {
        return;
    }

    const int64_t col = index % cols;
    const int64_t row = (index / cols) % rows;
    const int64_t batch = index / (rows * cols);
    const float *input_batch = input + batch * rows * inner;
    const float *other_batch = other + batch * inner * cols;
    float value = 0.0f;
    for (int64_t k = 0; k < inner; ++k) {
        value += input_batch[row * inner + k] * other_batch[k * cols + col];
    }
    output[index] = value;
}

__global__ void MatmulGradInputKernel(const float *other, const float *grad_output, float *grad_input,
                                      int64_t batch_count, int64_t rows, int64_t inner, int64_t cols) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch_count * rows * inner;
    if (index >= total) {
        return;
    }

    const int64_t k = index % inner;
    const int64_t row = (index / inner) % rows;
    const int64_t batch = index / (rows * inner);
    const float *other_batch = other + batch * inner * cols;
    const float *grad_output_batch = grad_output + batch * rows * cols;
    float value = 0.0f;
    for (int64_t col = 0; col < cols; ++col) {
        value += grad_output_batch[row * cols + col] * other_batch[k * cols + col];
    }
    grad_input[index] = value;
}

__global__ void MatmulGradOtherKernel(const float *input, const float *grad_output, float *grad_other,
                                      int64_t batch_count, int64_t rows, int64_t inner, int64_t cols) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch_count * inner * cols;
    if (index >= total) {
        return;
    }

    const int64_t col = index % cols;
    const int64_t k = (index / cols) % inner;
    const int64_t batch = index / (inner * cols);
    const float *input_batch = input + batch * rows * inner;
    const float *grad_output_batch = grad_output + batch * rows * cols;
    float value = 0.0f;
    for (int64_t row = 0; row < rows; ++row) {
        value += input_batch[row * inner + k] * grad_output_batch[row * cols + col];
    }
    grad_other[index] = value;
}

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

    const int threads_per_block = 256;
    const int64_t total = batch_count * rows * cols;
    const int num_blocks = (total + threads_per_block - 1) / threads_per_block;
    MatmulForwardKernel<<<num_blocks, threads_per_block>>>(static_cast<const float *>(input->DataPtr()),
                                                           static_cast<const float *>(other->DataPtr()),
                                                           static_cast<float *>(output->DataPtr()), batch_count, rows,
                                                           inner, cols);
    CUDA_CHECK(cudaGetLastError());
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
    CHECK(grad_output->GetDevice().IsCUDA());
    CHECK(grad_output->Dtype() == DataType::kFLOAT32);
    CHECK(grad_output->Dims() == output_dims);

    const int64_t batch_count = BatchCount(input_dims);
    const int64_t rows = input_dims[input_dims.size() - 2];
    const int64_t inner = input_dims.back();
    const int64_t cols = other_dims.back();
    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, input->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32, other->GetDevice());
    const int threads_per_block = 256;

    const int64_t input_elements = batch_count * rows * inner;
    const int input_blocks = (input_elements + threads_per_block - 1) / threads_per_block;
    MatmulGradInputKernel<<<input_blocks, threads_per_block>>>(
        static_cast<const float *>(other->DataPtr()), static_cast<const float *>(grad_output->DataPtr()),
        static_cast<float *>(grad_input->DataPtr()), batch_count, rows, inner, cols);

    const int64_t other_elements = batch_count * inner * cols;
    const int other_blocks = (other_elements + threads_per_block - 1) / threads_per_block;
    MatmulGradOtherKernel<<<other_blocks, threads_per_block>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<const float *>(grad_output->DataPtr()),
        static_cast<float *>(grad_other->DataPtr()), batch_count, rows, inner, cols);
    CUDA_CHECK(cudaGetLastError());
    return {grad_input, grad_other};
}

__global__ void BiasCopyKernel(float *output, const float *bias, int bs, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bs * out_features) {
        return;
    }
    int j = idx % out_features;
    output[idx] = bias[j];
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {

    /*
        !transpose: output = input * weight + bias
        output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]

        transpose:  output = input * weight^T + bias
        output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);

    // As for cublas:
    // C = alpha * op(B) * op(A) + beta * C
    // Dimensions:
    //   input:  (bs, in_features)
    //   weight: (in_features, out_features) or (out_features, in_features) if transposed
    //   output: (bs, out_features)
    const int64_t out_features = weight_dims[transpose ? 0 : 1];

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());

    if (bias) {
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], out_features);
        int threads_per_block = 256;
        int num_blocks = (bs * out_features + threads_per_block - 1) / threads_per_block;
        BiasCopyKernel<<<num_blocks, threads_per_block>>>(
            static_cast<float *>(output->DataPtr()), static_cast<const float *>(bias->DataPtr()), bs, out_features);
    } else {
        output->Fill<float>(0.0f);
    }

    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    if (transpose) {
        // weight is [out_features, in_features] here

        // output = input * weight.T --> output.T = weight * input.T
        // C = output.T[out_features, bs]
        // A = weight.T[in_features, out_features]
        // B = input.T[in_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, bs, in_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), in_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(output->DataPtr()), out_features));
    } else {
        // output = input * weight --> output.T =  weight.T * input.T
        // C = output.T[out_features, bs]
        // A = weight.T[out_features, in_features]
        // B = input.T[in_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, out_features, bs, in_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), out_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(output->DataPtr()), out_features));
    }
    CUBLAS_CHECK(cublasDestroy(handle));
    return output;
}

template <int BLOCK_SIZE>
__global__ void ReduceColumnsKernel(const float *__restrict__ input, float *__restrict__ output, int num_rows,
                                    int num_cols) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int row = blockIdx.x;
    float sum = 0.0f;

    for (int col = threadIdx.x; col < num_cols; col += blockDim.x) { sum += input[row * num_cols + col]; }

    float reduced = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        output[row] = reduced;
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, grad_output->GetDevice());
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32, grad_output->GetDevice());
    grad_input->Fill<float>(0.0f);
    grad_weight->Fill<float>(0.0f);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32,
                                             grad_output->GetDevice());
        grad_bias->Fill<float>(0.0f);
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    if (transpose) {
        // weight is [out_features, in_features] here

        // d_input = d_output * weight --> d_input.T = weight.T * d_output.T
        // C = d_input.T[in_features, bs]
        // A = weight.T[in_features, out_features]
        // B = d_output.T[out_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in_features, bs, out_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), in_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_input->DataPtr()), in_features));

        // d_weight = d_output.T * input --> d_weight.T = input.T * d_output
        // C = d_weight.T[in_features, out_features]
        // A = input.T[in_features, bs]
        // B = d_output.T[out_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, in_features, out_features, bs, &alpha,
                                 static_cast<const float *>(input->DataPtr()), in_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_weight->DataPtr()), in_features));
    } else {
        // weight is [in_features, out_features] here

        // d_input = d_output * weight.T --> d_input.T = weight * d_output.T
        // C = d_input.T[in_features, bs]
        // A = weight.T[out_features, in_features]
        // B = d_output.T[out_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, in_features, bs, out_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), out_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_input->DataPtr()), in_features));

        // d_weight = input.T * d_output --> d_weight.T = d_output.T * input
        // C = d_weight.T[out_features, in_features]
        // A = d_output.T[out_features, bs]
        // B = input.T[in_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, out_features, in_features, bs, &alpha,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(grad_weight->DataPtr()), out_features));
    }

    // d_bias = \sum_i(i=0, bs-1) d_output[i]
    if (bias) {
        constexpr int BLOCK_SIZE = 256;
        int threads_per_block = BLOCK_SIZE;
        int num_blocks = out_features;
        ReduceColumnsKernel<BLOCK_SIZE>
            <<<num_blocks, threads_per_block>>>(static_cast<const float *>(grad_output->DataPtr()),
                                                static_cast<float *>(grad_bias->DataPtr()), out_features, bs);
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_LINEAR_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_LINEAR_KERNEL(MatmulForward)
REGISTER_CUDA_LINEAR_KERNEL(MatmulBackward)
REGISTER_CUDA_LINEAR_KERNEL(LinearForward)
REGISTER_CUDA_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CUDA_LINEAR_KERNEL
