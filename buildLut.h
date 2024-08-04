//
// Created by yr on 24-7-31.
//

#ifndef FAST_BEV_BUILDLUT_H
#define FAST_BEV_BUILDLUT_H

#include <vector>
#include <memory>
#include <cmath>
#include "Tensor.h"


// 在CPU上构建查找表（LUT）的函数，将体积数据的体素映射到一系列图像上
// 参考文章：https://blog.csdn.net/qq_41204464/article/details/137571716
void build_LUT_CPU(std::vector<int32_t> n_voxels, Tensor voxel_size, Tensor origin,
                   Tensor projection, int32_t n_images, int32_t height, int32_t width, int32_t n_channels,
                   std::shared_ptr<int32_t>& LUT, std::shared_ptr<int32_t>& valid, std::shared_ptr<float>& volume) {

    // 投影矩阵的维度：6 x 3 x 4 (N, 3, 4)
    // 从n_voxels向量中提取X、Y、Z轴上的体素维度。
    int n_x_voxels = n_voxels[0];
    int n_y_voxels = n_voxels[1];
    int n_z_voxels = n_voxels[2];

    // 获取体素大小的指针，并提取具体尺寸
    float* voxel_sizep = (float*)voxel_size.get_data();
    float size_x = voxel_sizep[0];
    float size_y = voxel_sizep[1];
    float size_z = voxel_sizep[2];

    // 获取原点位置的指针，并提取具体坐标
    float* originp = (float*)origin.get_data();
    float origin_x = originp[0];
    float origin_y = originp[1];
    float origin_z = originp[2];

    // 计算总体素数
    int nrof_voxels = n_x_voxels * n_y_voxels * n_z_voxels;

    // 从智能指针获取查找表（LUT）和有效标记数组的原始指针
    int32_t* LUTp = LUT.get();
    int32_t* validp = valid.get();

    // 辅助向量，用于计算和存储转换结果
    std::vector<float> ar(3);
    std::vector<float> pt(3);
    size_t offset = 0;  // LUT中的偏移量
    float count = 0.0;  // 用于计数

    // 遍历每个体素
    for (int zi = 0; zi < n_z_voxels; ++zi) {
        for (int yi = 0; yi < n_y_voxels; ++yi) {
            for (int xi = 0; xi < n_x_voxels; ++xi) {
                auto current_lut = &LUTp[offset * 2];
                *current_lut = -1;  // 初始设置为-1，表示无效映射
                *(current_lut + 1) = 0;  // 第二个值用于存储映射信息，初始为0
                // 遍历每个图像，尝试找到当前体素在这些图像中的映射
                for (int img = 0; img < n_images; img++) {
                    // 计算体素在世界坐标系中的位置
                    pt[0] = (xi - n_x_voxels / 2.0f) * size_x + origin_x;
                    pt[1] = (yi - n_y_voxels / 2.0f) * size_y + origin_y;
                    pt[2] = (zi - n_z_voxels / 2.0f) * size_z + origin_z;

                    // 使用投影矩阵将体素坐标转换为图像坐标
                    for (int i = 0; i < 3; ++i) {
                        ar[i] = ((float*)projection.get_data())[((img * 3) + i) * 4 + 3];
                        for (int j = 0; j < 3; ++j) {
                            ar[i] += ((float*)projection.get_data())[(img * 3 + i) * 4 + j] * pt[j];
                        }
                    }

                    // 计算投影后的图像坐标
                    int x = round(ar[0] / ar[2]);
                    int y = round(ar[1] / ar[2]);
                    float z = ar[2];

                    // 检查图像坐标是否有效
                    if ((x >= 0) && (y >= 0) && (x < width) && (y < height) && (z > 0)) {
                        *current_lut = img;  // 记录图像索引
                        *(current_lut + 1) = y * width + x;  // 记录图像内的具体位置
                        count+=1;

                        validp[offset] = 1;  // 标记为有效
                        break;  // 找到有效映射后停止搜索
                    }
                }
                ++offset;  // 处理下一个体素
            }
        }
    }
}

#endif //FAST_BEV_BUILDLUT_H
