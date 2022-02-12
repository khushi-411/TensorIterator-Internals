cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(examples)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGES "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGES}")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lrt -lpthread -lresolv")

add_executable(examples examples.cpp)
target_link_libraries(examples "${TORCH_LIBRARIES}")
set_property(TARGET examples PROPERTY CXX_STANDARD 14)
