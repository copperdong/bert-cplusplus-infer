//
// Created by 刘佳伟 on 2019/6/27.
//

#ifndef FUPU_PROJECT_BERT_H
#define FUPU_PROJECT_BERT_H

#include "tensorflow/core/public/session.h"

#include "base_type.h"
#include "tokenization.h"

using tensorflow::Session;
using tensorflow::Tensor;

class Bert {
private:
    void ReadTfModel(std::string ckpt_path);

    // tensorflow的session对象指针
    Session *session_;
    // 设置sess的运行线程数
    int num_threads_;
    // 设置tensorflow的op计算的设备
    std::string device_str_;
    //
    FullTokenizer *ft_;
    //
    int max_seq_length_;
    //
    std::string pooling_strategy_;
    //
    std::vector<int> pooling_layer_;

public:
    Bert(std::string ckpt_path,
         std::string vocab_path,
         int num_threads,
         std::vector<int> pooling_layer,
         std::string pooling_strategy,
         std::string device_str,
         int max_seq_length = 128);

    ~Bert() {
        session_->Close();
        delete ft_;
    }

    void Encoding(std::vector <Example> &inputs, Mat2d<float> &seq_outputs);
};

#endif //FUPU_PROJECT_BERT_H
