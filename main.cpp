#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cassert>

class Tensor {
public:
    Tensor(std::vector<float> data, std::vector<int> shape) : data_(data), shape_(shape) {}

    float* get_data() {
        return data_.data();  //
    }

    std::vector<int> shape() {
        return shape_;
    }

private:
    std::vector<float> data_;
    std::vector<int> shape_;
};


/**
 * @brief 构建查找表 (Look-Up Table, LUT) 以便将体素坐标映射到图像坐标
 * 参考文章：https://blog.csdn.net/qq_41204464/article/details/137571716(fastbev)
 *         https://blog.csdn.net/MinyounZhang/article/details/134582017(lss)
 *
 * @param n_voxels   包含体素网格在每个方向上的数量（X, Y, Z）
 * @param voxel_size 每个体素的大小 (Tensor)
 * @param origin     BEV网格的原点坐标 (Tensor)
 * @param projection 投影矩阵，用于将体素坐标转换为图像坐标 (Tensor)
 * @param n_images   图像的数量
 * @param height     图像的高度（像素）
 * @param width      图像的宽度（像素）
 * @param n_channels 图像的通道数量
 * @param LUT        查找表，用于存储体素到图像的映射（输出参数）
 * @param valid      标志数组，用于指示哪些体素有效（输出参数）
 * @param volume     存储体素值的数组（输出参数）
 */
void build_LUT_CPU(std::vector<int32_t> n_voxels, Tensor voxel_size, Tensor origin,
                   Tensor projection, int32_t n_images, int32_t height, int32_t width, int32_t n_channels,
                   std::shared_ptr<int32_t>& LUT, std::shared_ptr<int32_t>& valid, std::shared_ptr<float>& volume) {
    // BEV特征在x,y,z方向上每个方向上的网格数量
    int n_x_voxels = n_voxels[0];
    int n_y_voxels = n_voxels[1];
    int n_z_voxels = n_voxels[2];

    // 获取每个方向上的网格尺寸大小
    float* voxel_sizep = (float*)voxel_size.get_data();
    float size_x = voxel_sizep[0];
    float size_y = voxel_sizep[1];
    float size_z = voxel_sizep[2];

    // 获取BEV网格原始坐标
    float* originp = (float*)origin.get_data();
    float origin_x = originp[0];
    float origin_y = originp[1];
    float origin_z = originp[2];

    int nrof_voxels = n_x_voxels * n_y_voxels * n_z_voxels;

    // 为什么要使用get()来重新定义变量？?
    //
    int32_t* LUTp = LUT.get();  // LUT.get() 返回的是指向数组的指针，
    int32_t* validp = valid.get();

    std::vector<float> ar(3);
    std::vector<float> pt(3);
    size_t offset = 0;
    float count = 0.0;

    for (int zi = 0; zi < n_z_voxels; ++zi) {
        for (int yi = 0; yi < n_y_voxels; ++yi) {
            for (int xi = 0; xi < n_x_voxels; ++xi) {
                // 这里什么意思？？
                // 回答：current_lut是一个指针，指向LUTp数组中的offset*2位置，这里的2是因为每个体素对应两个值，一个是图像索引，一个是像素索引
                auto current_lut = &LUTp[offset * 2];
                *current_lut = -1;        // 存储图像索引
                *(current_lut + 1) = 0;   // 存储对应图像中的像素索引
                for (int img = 0; img < n_images; img++) {
                    pt[0] = (xi - n_x_voxels / 2.0f) * size_x + origin_x;
                    pt[1] = (yi - n_y_voxels / 2.0f) * size_y + origin_y;
                    pt[2] = (zi - n_z_voxels / 2.0f) * size_z + origin_z;

                    // 用投影矩阵将体素坐标转换为图像坐标，i表示投影矩阵的行，j表示投影矩阵的列
                    // 对一个维度为[d1,d2,d3...dn]的数组，索引[i1,i2,i3...in]在内存中索引的计算公式为：
                    // idex = i1*d2*d3*...*dn + i2*d3*...*dn + i3*d4*...*dn + ... + in
                    // 注：第一次，可以把d1看作行维度，d2,d3...dn看作列维度，第二次，可以把d2看作行，d3,d4...dn看作列，以此类推
                    // 第一次，索引为i1*d2*d3...dn + 列索引
                    // 第二交，需要计算第一次剩下的列索引，此时又看作一个[d2,d3...dn]的数组，所以列索引为i2*d3*...*dn + 列索引
                    // 倒数第二次，索引为i(n-1)*dn + 列索引
                    // 最后一次，索引为in
                    // 所以，最终在内存中的索引为i1*d2*d3*...*dn + i2*d3*...*dn + i3*d4*...*dn + ... + in
                    // 举例：projection的维度是(N, 3, 4)，则对于第img个图像，投影矩阵的第i行第j列的元素在内存中的索引为
                    // img*3*4 + i*4 + j = (img*3 + i)*4 + j，对应上了代码中的((img * 3) + i) * 4 + j
                    for (int i = 0; i < 3; ++i) {
                        // ar[i] = 投影矩阵的第i行第4列
                        // projection的维度是(N, 3, 4)，projection.get_data()返回的是指向数组的指针
                        // projection在内存中的
                        ar[i] = ((float*)projection.get_data())[((img * 3) + i) * 4 + 3];
                        for (int j = 0; j < 3; ++j) {
                            // 将projection中的i行j列元素和体素坐标点第j个元素相乘，对应世界坐标转换到像素坐标的矩阵乘法第j个元素结果
                            ar[i] += ((float*)projection.get_data())[(img * 3 + i) * 4 + j] * pt[j];
                        }
                    }

                    // 用深度Z对投影后的坐标进行归一化处理，得到最终的像素坐标。参考视觉slam十四讲的针孔相机投影模型
                    // TODO 这里应该还要除以一个特征图对应原图的下采样倍数，才对应特征图上的坐标
                    int x = round(ar[0] / ar[2]);
                    int y = round(ar[1] / ar[2]);
                    float z = ar[2];

                    // 判断投影后的像素坐标x,y是否在图像内，且z是否在摄像头前方，如果满足，则认为找到了BEV特征中当前坐标(xi,yi,zi)
                    // 在2D特征对应的坐标(x,y,z)。
                    // 注：一个点的三维空间坐标(xi,yi,zi)映射加像素坐标系中后的坐标(x,y,zi)。深度值是相等的，可以参考相机模型投影公式
                    if ((x >= 0) && (y >= 0) && (x < width) && (y < height) && (z > 0)) {
                        *current_lut = img;
                        // 这里得到的是当前体素(xi,yi,zi)在第img个图像特征图中的索引，将x,y方向上的所有格子看作一个一维数组
                        *(current_lut + 1) = y * width + x;
                        count += 1;

                        validp[offset] = 1;
                        break;
                    }
                }
                ++offset;
            }
        }
    }
}

void test_build_LUT_CPU() {
    std::vector<int32_t> n_voxels = {2, 2, 2};
    std::vector<float> voxel_size_data = {1.0f, 1.0f, 1.0f};
    std::vector<float> origin_data = {0.0f, 0.0f, 0.0f};
    std::vector<float> projection_data = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 1.0f
    };

    int32_t n_images = 1;
    int32_t height = 2;
    int32_t width = 2;
    int32_t n_channels = 1;

    Tensor voxel_size(voxel_size_data, {3});
    Tensor origin(origin_data, {3});
    Tensor projection(projection_data, {1, 3, 4});

    size_t n_voxels_total = n_voxels[0] * n_voxels[1] * n_voxels[2];
    std::shared_ptr<int32_t> LUT(new int32_t[n_voxels_total * 2], std::default_delete<int32_t[]>());
    std::shared_ptr<int32_t> valid(new int32_t[n_voxels_total], std::default_delete<int32_t[]>());
    std::shared_ptr<float> volume(new float[n_voxels_total], std::default_delete<float[]>());

    for (size_t i = 0; i < n_voxels_total * 2; ++i) LUT.get()[i] = -1;
    for (size_t i = 0; i < n_voxels_total; ++i) valid.get()[i] = 0;

    build_LUT_CPU(n_voxels, voxel_size, origin, projection, n_images, height, width, n_channels, LUT, valid, volume);

    std::cout << "LUT:\n";
    for (size_t i = 0; i < n_voxels_total; ++i) {
        std::cout << LUT.get()[i * 2] << " " << LUT.get()[i * 2 + 1] << "\n";
    }

    std::cout << "\nValid:\n";
    for (size_t i = 0; i < n_voxels_total; ++i) {
        std::cout << valid.get()[i] << " ";
    }
    std::cout << std::endl;

    // Example assertions
    assert(LUT.get()[0] == 0);
    assert(LUT.get()[1] == 0);
    assert(valid.get()[0] == 1);

    // Add further assertions as needed
}

int main() {
    test_build_LUT_CPU();
    std::cout << "All tests passed!" << std::endl;
    return 0;
    // merge test
}
