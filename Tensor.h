//
// Created by yr on 24-7-31.
//

#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>

class Tensor {
public:
    // 构造函数，使用维度和数据初始化Tensor
    Tensor(const std::vector<int>& dims, const std::vector<float>& data)
            : dims_(dims), data_(data) {
        if (dims.empty() || data.empty()) {
            throw std::invalid_argument("Dimensions and data cannot be empty");
        }

        // 计算总元素数并检查数据大小是否匹配
        int total_size = 1;
        for (int dim : dims) {
            total_size *= dim;
        }
        if (total_size != data.size()) {
            throw std::invalid_argument("Data size does not match dimensions");
        }
    }

    // 获取数据的指针
    const float* get_data() const {
        return data_.data();
    }

    // 获取Tensor的维度
    const std::vector<int>& get_dims() const {
        return dims_;
    }

private:
    std::vector<int> dims_;  // 存储Tensor的维度
    std::vector<float> data_;  // 存储Tensor的数据
};

#endif // TENSOR_H


