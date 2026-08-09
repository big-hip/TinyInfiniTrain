#pragma once

#include <iostream>
#include <map>
#include <type_traits>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/device.h"

namespace infini_train {
class KernelFunction {
public:
    template <typename FuncT> explicit KernelFunction(FuncT &&func) : func_ptr_(reinterpret_cast<void *>(func)) {}

    template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
        using FuncT = RetT (*)(ArgsT...);
        CHECK_NE(func_ptr_, nullptr);
        return reinterpret_cast<FuncT>(func_ptr_)(args...);
    }

private:
    void *func_ptr_ = nullptr;
};

class Dispatcher {
public:
    using KeyT = std::pair<DeviceType, std::string>;

    static Dispatcher &Instance() {
        static Dispatcher instance;
        return instance;
    }

    const KernelFunction &GetKernel(KeyT key) const {
        CHECK(key_to_kernel_map_.contains(key))
            << "Kernel not found: " << key.second << " on device: " << static_cast<int>(key.first);
        return key_to_kernel_map_.at(key);
    }

    template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
        CHECK(!key_to_kernel_map_.contains(key)) << "Kernel already registered: " << key.second
                                                  << " on device: " << static_cast<int>(key.first);
        key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
    }

private:
    std::map<KeyT, KernelFunction> key_to_kernel_map_;
};
} // namespace infini_train

#define INFINI_TRAIN_CONCAT_IMPL(x, y) x##y
#define INFINI_TRAIN_CONCAT(x, y) INFINI_TRAIN_CONCAT_IMPL(x, y)

#define REGISTER_KERNEL(device, kernel_name, kernel_func)                                                              \
    static const bool INFINI_TRAIN_CONCAT(_infini_train_kernel_registered_, __COUNTER__) = [] {                      \
        infini_train::Dispatcher::Instance().Register({device, #kernel_name}, kernel_func);                          \
        return true;                                                                                                  \
    }();
