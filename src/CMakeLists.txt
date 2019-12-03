aux_source_directory(../third_party/unilib/unilib unilib_src)
#add_executable(token_test
#        ./tokenization.cpp
#        ${unilib_src}
#)
#target_link_libraries(
#        token_test
#        libboost_locale.so
#)


add_executable(bert_test
        ./bert.cpp
        ./tokenization.cpp
        ${unilib_src})

target_link_libraries(
        bert_test
        libtensorflow_framework.so
        libtensorflow_cc.so
        libtensorflow-core.a
        libprotoc.a
        libprotobuf.a
)