//
// Created by 刘佳伟 on 2019/6/27.
//

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "bert.h"

using tensorflow::Status;

void ConvertExamplesToFeatures(std::vector <Example> &examples,
                               int max_seq_length,
                               FullTokenizer &tokenizer,
                               std::vector <Features> &examples_features) {
    int example_nums = examples.size();
    for (int i = 0; i < example_nums; i++) {
        Features temp_features;
        std::vector <std::wstring> tokens_a;
        tokenizer.Tokenize(examples[i].text_a, tokens_a);
        std::vector <std::wstring> tokens_b;
        if (!examples[i].text_b.empty()) {
            tokenizer.Tokenize(examples[i].text_b, tokens_b);
        }


        if (!tokens_b.empty()) {
            int total_length;
            while (true) {
                total_length = tokens_a.size() + tokens_b.size();
                if (total_length <= max_seq_length - 3) break;
                if (tokens_a.size() > tokens_b.size())
                    tokens_a.pop_back();
                else
                    tokens_b.pop_back();
            }
        } else {
            if (tokens_a.size() > max_seq_length - 2) {
                tokens_a.pop_back();
                tokens_a.pop_back();
            }
        }

        temp_features.tokens.push_back(L"[CLS]");
        temp_features.input_type_ids.push_back(0);
        int tokens_a_size = tokens_a.size();
        for (int i = 0; i < tokens_a_size; i++) {
            temp_features.tokens.push_back(tokens_a[i]);
            temp_features.input_type_ids.push_back(0);
        }
        temp_features.tokens.push_back(L"[SEP]");
        temp_features.input_type_ids.push_back(0);

        if (!tokens_b.empty()) {
            int tokens_b_size = tokens_b.size();
            for (int i = 0; i < tokens_b_size; i++) {
                temp_features.tokens.push_back(tokens_b[i]);
                temp_features.input_type_ids.push_back(1);
            }
            temp_features.tokens.push_back(L"[SEP]");
            temp_features.input_type_ids.push_back(1);
        }

        tokenizer.ConvertTokensToIds(temp_features.tokens, temp_features.input_ids);

        int input_ids_nums = temp_features.input_ids.size();
        for (int i = 0; i < input_ids_nums; i++) {
            temp_features.input_mask.push_back(1);
        }

        //zero_padding

        while (temp_features.input_ids.size() < max_seq_length) {
            temp_features.input_ids.push_back(0);
            temp_features.input_mask.push_back(0);
            temp_features.input_type_ids.push_back(0);
        }

        examples_features.push_back(temp_features);
    }


    for (int i = 0; i < examples_features.size(); i++) {
        for (int j = 0; j < examples_features[i].input_ids.size(); j++) {
            std::cout << examples_features[i].input_ids[j] << " ";
        }
        std::cout << std::endl;
    }
}


void FeaturesToTensor(std::vector <Features> &features,
                      tensorflow::Tensor &input_id_tensor,
                      tensorflow::Tensor &input_mask_tensor,
                      tensorflow::Tensor &input_type_id_tensor) {
    auto plane_tensor_input_id = input_id_tensor.tensor<int, 2>();
    auto plane_tensor_input_mask = input_mask_tensor.tensor<int, 2>();
    auto plane_tensor_input_type_id = input_type_id_tensor.tensor<int, 2>();
    int batch_size = features.size();
    int seq_length = features[0].input_ids.size();

    std::cout << "batch_size is" << batch_size << "seq_length is " << seq_length << std::endl;

    for (int i = 0; i < batch_size; i++)
        for (int j = 0; j < seq_length; j++) {
            plane_tensor_input_id(i, j) = features[i].input_ids[j];
            plane_tensor_input_mask(i, j) = features[i].input_mask[j];
            plane_tensor_input_type_id(i, j) = features[i].input_type_ids[j];
        }
}


void TensorToNdarray(tensorflow::Tensor &input,
                     Mat2d<float> &output) {
    tensorflow::TensorShape tensor_shape = input.shape();
    if (tensor_shape.dims() != 2) {
        std::cout << "The output tensor must be with 2 dims including batch_size, output_dim" << std::endl;
        return;
    }

    auto tensor_mapped = input.tensor<float, 2>();
    int batch_size = tensor_shape.dim_size(0),
            hidden_size = tensor_shape.dim_size(1);

    for (int i = 0; i < batch_size; i++) {
        Mat1d<float> hidden_vec;
        for (int j = 0; j < hidden_size; j++) {
            hidden_vec.push_back(tensor_mapped(i, j));
        }
        output.push_back(hidden_vec);
    }
}

void TensorToEigenMat3d(std::vector <tensorflow::Tensor> &input, EigenMat3d<float> &output) {
    tensorflow::TensorShape tensor_shape = input[0].shape();
    int batch_size = tensor_shape.dim_size(0);
    int num_frames = tensor_shape.dim_size(1);
    int output_dim = tensor_shape.dim_size(2);
    int tensor_nums = input.size();

    std::cout << "batch_size: " << batch_size << std::endl;


    for (int i = 0; i < batch_size; i++) {
        EigenMat2d<float> temp_mat(num_frames, output_dim * tensor_nums);
        for (int j = 0; j < tensor_nums; j++) {
            auto temp_mapped = input[j].tensor<float, 3>();
            for (int k = 0; k < num_frames; k++)
                for (int l = 0; l < output_dim; l++) {
                    temp_mat(k, l + j * output_dim) = temp_mapped(i, k, l);
                }
        }
        output.push_back(temp_mat);
    }

    std::cout << output[0].rows() << "  " << output[0].cols() << std::endl;
}


Bert::Bert(std::string ckpt_path,
           std::string vocab_path,
           int num_threads,
           std::vector<int> pooling_layer,
           std::string pooling_strategy,
           std::string device_str,
           int max_seq_length) {
    num_threads_ = num_threads;
    device_str_ = device_str;
    max_seq_length_ = max_seq_length;
    pooling_layer_.insert(pooling_layer_.end(), pooling_layer.begin(), pooling_layer.end());
    pooling_strategy_ = pooling_strategy;
    ft_ = new FullTokenizer(vocab_path);
    ReadTfModel(ckpt_path);
    std::cout << "bert encoding init success!" << std::endl;
}


void Bert::ReadTfModel(std::string ckpt_path) {
    std::string meta_graph_path = ckpt_path + ".meta";
    //session option 设置
    tensorflow::SessionOptions session_options;
    session_options.config.set_use_per_session_threads(true);
    session_options.config.set_intra_op_parallelism_threads(num_threads_);
    session_options.config.set_inter_op_parallelism_threads(num_threads_);
    session_options.config.mutable_gpu_options()->set_allow_growth(true);
    session_options.config.set_allow_soft_placement(true);


    Status status = tensorflow::NewSession(session_options, &session_);
    if (!status.ok()) {
        std::cout << status.ToString();
        return;
    }

    //加载图定义
    tensorflow::MetaGraphDef graph_def;
    status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), meta_graph_path, &graph_def);
    if (!status.ok()) {
        std::cout << status.ToString();
        return;
    }
    //tensorflow::graph::SetDefaultDevice(device_str_, &(graph_def.graph_def()));

    status = session_->Create(graph_def.graph_def());
    if (!status.ok()) {
        std::cout << status.ToString();
        return;
    }

    tensorflow::Tensor checkpointPathTensor(tensorflow::DT_STRING, tensorflow::TensorShape());
    checkpointPathTensor.scalar<std::string>()() = ckpt_path;

    status = session_->Run(
            {{graph_def.saver_def().filename_tensor_name(), checkpointPathTensor}},
            {},
            {graph_def.saver_def().restore_op_name()},
            nullptr);

    if (!status.ok()) {
        std::cout << status.ToString();
        return;
    }
}


void Bert::Encoding(std::vector <Example> &inputs, Mat2d<float> &seq_outputs) {
    std::vector <Features> features;
    ConvertExamplesToFeatures(inputs, max_seq_length_, *ft_, features);
    std::cout << "convert examples to features and length is " << features.size() << std::endl;

    int batch_size = features.size();
    int seq_length = features[0].input_ids.size();

    tensorflow::Tensor input_id_tensor(tensorflow::DT_INT32, {batch_size, seq_length});
    tensorflow::Tensor input_mask_tensor(tensorflow::DT_INT32, {batch_size, seq_length});
    tensorflow::Tensor input_type_id_tensor(tensorflow::DT_INT32, {batch_size, seq_length});

    FeaturesToTensor(features, input_id_tensor, input_mask_tensor, input_type_id_tensor);

    std::cout << "features to tensor success" << std::endl;

    std::vector <std::pair<std::string, tensorflow::Tensor>> feed_inputs = {
            {"Placeholder:0",   input_id_tensor},
            {"Placeholder_1:0", input_mask_tensor},
            {"Placeholder_2:0", input_type_id_tensor}
    };

    std::vector <tensorflow::Tensor> output;

    //,
    Status status = session_->Run(feed_inputs,
                                  {
                                          "bert/pooler/Squeeze:0",
                                          "bert/encoder/Reshape_13",
                                          "bert/encoder/Reshape_12",
                                          "bert/encoder/Reshape_11",
                                          "bert/encoder/Reshape_10",
                                          "bert/encoder/Reshape_9",
                                          "bert/encoder/Reshape_8",
                                          "bert/encoder/Reshape_7",
                                          "bert/encoder/Reshape_6",
                                          "bert/encoder/Reshape_5",
                                          "bert/encoder/Reshape_4",
                                          "bert/encoder/Reshape_3",
                                          "bert/encoder/Reshape_2"},
                                  {},
                                  &output);

    if (!status.ok()) {
        std::cout << status.ToString();
        return;
    }

    /*

std::vector <tensorflow::Tensor> tensor_to_concat;
for (int i = 0; i < pooling_layer_.size(); i++)
    tensor_to_concat.push_back(output[i]);

EigenMat3d<float> concat_eigen;
TensorToEigenMat3d(tensor_to_concat, concat_eigen);

//TODO(jiawei): 看看有没有什么更加高效的方案
switch (pooling_layer_.size()) {
    case 0:
        std::cout << "must specify pooling layer";
        return;
    case 1:
        break;
    case 2:
        break;
    case 3:
        break;
    case 4:
        break;
    case 5:
        break;
    case 6:
        break;
    case 7:
        break;
    case 8:
        break;
    case 9:
        break;
    case 10:
        break;
    case 11:
        break;
    case 12:
        break;
    default:
        std::cout << "must specify pooling layer";
        return;
}
 */

    //TensorToNdarray(output[0], seq_outputs);
}


int main() {
    std::wstring test_str = L"I like my mother";
    std::vector <Example> inputs;
    Example e;
    e.text_a = test_str;
    inputs.push_back(e);
    //inputs.push_back(e);
    std::string ckpt_path = "/data/liujiawei/bert-model/uncased_L-12_H-768_A-12/bert_model.ckpt";
    std::string device_str = "/cpu:0";
    std::string vocab_path = "/data/liujiawei/bert-model/uncased_L-12_H-768_A-12/vocab.txt";

    std::vector<int> pooling_layer = {1};
    Bert bt(ckpt_path, vocab_path, 2, pooling_layer, "first_token", device_str);

    Mat2d<float> result;
    bt.Encoding(inputs, result);

    int batch_size = result.size();
    int dim_size = result[0].size();

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < dim_size; j++) {
            std::cout << result[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}