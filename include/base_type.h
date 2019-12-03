//
// Created by 刘佳伟 on 2019/6/20.
//

#ifndef FUPU_PROJECT_BASE_TYPE_H
#define FUPU_PROJECT_BASE_TYPE_H

#include <unordered_map>
#include <string>
#include <Eigen/Core>

using vocab_map = std::unordered_map<std::wstring, int>;
using inv_vocab_map = std::unordered_map<int, std::wstring>;

template<typename T>
using Mat1d = std::vector<T>;

template<typename T>
using Mat2d = std::vector<std::vector<T>>;

template<typename T>
using Mat3d = std::vector<std::vector<std::vector < T>>>;

template <typename T>
using EigenMat2d = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using EigenMat3d = std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;


struct Example{
    std::wstring text_a;
    std::wstring text_b;
};


struct Features{
    std::vector<std::wstring> tokens;
    std::vector<int> input_ids;
    std::vector<int> input_mask;
    std::vector<int> input_type_ids;
};

#endif //FUPU_PROJECT_BASE_TYPE_H
