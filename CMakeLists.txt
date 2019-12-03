cmake_minimum_required(VERSION 3.12)

project(bert-as-service)
SET(CMAKE_C_COMPILER /usr/local/gcc-5.4.0/bin/gcc)
SET(CMAKE_CXX_COMPILER /usr/local/gcc-5.4.0/bin/g++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic -g")


set(BOOST_DIR /data/liujiawei/home_backup/github/boost_1_70_0)
set(TENSORFLOW_DIR /data/liujiawei/github/kaldi/tools/extras/tensorflow)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

include_directories(
        ${BOOST_DIR}
        ${TENSORFLOW_DIR}
        ${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/proto
        ${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/protobuf-host/include
        ${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/eigen
        ${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/absl
        ${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/nsync/public
        ${TENSORFLOW_DIR}/bazel-genfiles
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/third_party/unilib/unilib
)


link_directories(
        #${BOOST_DIR}/lib
        ${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/lib
        ${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/protobuf-host/lib
        #/data/liujiawei/home_backup/github/grpc/build/third_party/protobuf
        ${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/nsync/builds/default.linux.c++11
        ${TENSORFLOW_DIR}/bazel-bin/tensorflow
)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

ADD_SUBDIRECTORY(src)
