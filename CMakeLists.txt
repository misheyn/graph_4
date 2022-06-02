cmake_minimum_required(VERSION 3.22)
project(graph_4 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(graph_4 main.cu)

target_link_libraries(graph_4 -lglut -lGL -lGLU)

set_target_properties(graph_4 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
