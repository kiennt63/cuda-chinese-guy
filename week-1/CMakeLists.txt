add_library(vecadd lecture1_5.cu)
set_target_properties(vecadd PROPERTIES CUDA_ARCHITECTURES "72")

add_executable(test main.cpp)

target_link_libraries(test PUBLIC
    vecadd
)

add_executable(matrix_mult matrix_mult.cu)
set_target_properties(matrix_mult PROPERTIES CUDA_ARCHITECTURES "72")
