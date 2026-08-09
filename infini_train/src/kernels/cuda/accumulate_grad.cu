#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

__global__ void AccumulateGradKernel(const float *grad_ptr, float rate, float *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += rate * grad_ptr[idx];
    }
}

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    size_t num_elements = gradient->NumElements();

    const float *grad_ptr = static_cast<const float *>(gradient->DataPtr());
    float *tensor_ptr = static_cast<float *>(tensor->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, rate, tensor_ptr, num_elements);
}

__global__ void AdamAccumulateGradKernel(const float *grad, float *param, float *m, float *v, size_t num_elements,
                                         float learning_rate, float beta1, float beta2, float eps,
                                         float beta1_correction, float beta2_correction) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) {
        return;
    }

    const float gradient = grad[idx];
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * gradient;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * gradient * gradient;
    const float m_hat = m[idx] / beta1_correction;
    const float v_hat = v[idx] / beta2_correction;
    param[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + eps);
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    CHECK(grad->GetDevice() == param->GetDevice());
    CHECK(m->GetDevice() == param->GetDevice());
    CHECK(v->GetDevice() == param->GetDevice());
    CHECK(static_cast<int>(grad->Dtype()) == static_cast<int>(DataType::kFLOAT32));
    CHECK(static_cast<int>(param->Dtype()) == static_cast<int>(DataType::kFLOAT32));
    CHECK(static_cast<int>(m->Dtype()) == static_cast<int>(DataType::kFLOAT32));
    CHECK(static_cast<int>(v->Dtype()) == static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(grad->NumElements(), param->NumElements());
    CHECK_EQ(m->NumElements(), param->NumElements());
    CHECK_EQ(v->NumElements(), param->NumElements());
    CHECK_GT(t, 0);

    const float beta1_correction = 1.0f - powf(beta1, static_cast<float>(t));
    const float beta2_correction = 1.0f - powf(beta2, static_cast<float>(t));
    const size_t num_elements = param->NumElements();
    const int threads_per_block = 256;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    AdamAccumulateGradKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const float *>(grad->DataPtr()), static_cast<float *>(param->DataPtr()),
        static_cast<float *>(m->DataPtr()), static_cast<float *>(v->DataPtr()), num_elements, learning_rate, beta1,
        beta2, eps, beta1_correction, beta2_correction);
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
