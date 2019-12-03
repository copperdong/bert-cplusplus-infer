## Bert-as-service++



### Build & Deploy 
* how to build
    ```bash
    #下载tensorflow
    git clone https://github.com/tensorflow/tensorflow
    #进入tensorflow文件夹下
    cd tensorflow
    #配置tensorflow安装选项
    ./configure
    
    #编译C++ API，生成.so文件，Tensorflow调用CUDA
    bazel build --config=opt --config=cuda //tensorflow:libtensorflow_cc.so
    
    #编译C++ API，生成.so文件，Tensorflow不调用CUDA
    bazel build --config=opt //tensorflow:libtensorflow_cc.so
    
    进入tensorflow主目录
    cd tensorflow
    运行编译第三方库的脚本
    ./tensorflow/contrib/makefile/build_all_linux.sh
    
    ```

### Others
* use [spdlog](https://github.com/gabime/spdlog) log system
* use [catch2](https://github.com/catchorg/Catch2) unit test framework
* use [gRPC](https://github.com/grpc/grpc) for service
* use [yaml](https://github.com/jbeder/yaml-cpp) for config
* use [json](https://github.com/nlohmann/json) for 
* FUTURE dynamic loaded config (log or other things)
