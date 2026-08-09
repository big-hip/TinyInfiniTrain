#include <cstddef>
#include <cmath>
#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
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

    const float beta1_correction = 1.0f - std::pow(beta1, static_cast<float>(t));
    const float beta2_correction = 1.0f - std::pow(beta2, static_cast<float>(t));
    const auto *grad_ptr = static_cast<const float *>(grad->DataPtr());
    auto *param_ptr = static_cast<float *>(param->DataPtr());
    auto *m_ptr = static_cast<float *>(m->DataPtr());
    auto *v_ptr = static_cast<float *>(v->DataPtr());
    for (int64_t idx = 0; idx < param->NumElements(); ++idx) {
        const float gradient = grad_ptr[idx];
        m_ptr[idx] = beta1 * m_ptr[idx] + (1.0f - beta1) * gradient;
        v_ptr[idx] = beta2 * v_ptr[idx] + (1.0f - beta2) * gradient * gradient;
        const float m_hat = m_ptr[idx] / beta1_correction;
        const float v_hat = v_ptr[idx] / beta2_correction;
        param_ptr[idx] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
