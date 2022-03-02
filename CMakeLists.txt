cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(rotst VERSION 1.0)

file(GLOB_RECURSE SRC_FILES library/src/*.cpp external/lean_vtk/*.cpp)

add_executable(rotst drivers/main.cpp ${SRC_FILES})

target_include_directories(rotst PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/library/inc ${CMAKE_CURRENT_SOURCE_DIR}/external/lean_vtk ${CMAKE_CURRENT_SOURCE_DIR}/../eigen)
