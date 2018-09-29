#====================
cmake_minimum_required (VERSION 3.9)
PROJECT(ICP_Demo LANGUAGES C CXX CUDA)

# Useful CMake options
#set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.2/bin/nvcc")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_FLAGS "-std=c++11 -fpermissive")
set(OpenGL_GL_PREFERENCE "GLVND")
set(CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=sm_61")
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

# Search desired Qt packages
#find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)
find_package(glm REQUIRED)
message(STATUS "GLM included at ${GLM_INCLUDE_DIR}")
find_package(Boost COMPONENTS filesystem REQUIRED)  #only need this till nvcc starts supporting c++17 compiler

set(GLAD_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/glad/include")
set(GLAD_SRC "${GLAD_INCLUDE_DIR}/../src/glad.c")
add_library("GladLib" ${GLAD_SRC})
target_include_directories("GladLib" PRIVATE "${GLAD_INCLUDE_DIR}")

set(CUHELPER_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/cuda_helper")
set(INCLUDE_DIRS  ${INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${CUHELPER_INCLUDE_DIR} ${GLM_INCLUDE_DIR} ${OPENGL_INCLUDE_DIRS} ${SDL2_INCLUDE_DIRS} ${GLAD_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OPENGL_LIBRARY} ${CUDA_LIBRARIES} ${SDL2_LIBRARIES} ${Boost_LIBRARIES} ${GladLib})

#specify include directory
#include_directories(/usr/include)
#include_directories(${CMAKE_SOURCE_DIR} )
include_directories(${INCLUDE_DIRS})

set(SOURCES DepthMain.cpp camera.cpp Window.cpp Application.cpp prereq.h ShaderProgram.hpp Frustum.cpp camera.h Window.h
    Application.h Frustum.h  ${GLAD_SRC})

#Compile
add_library(CameraTracking STATIC CameraTracking.cu CameraTracking.h ${CUHELPER_INCLUDE_DIR})
#target_include_directories(CameraTracking STATIC CameraTracking.h "${CUHELPER_INCLUDE_DIR}")
set_target_properties(CameraTracking PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(CameraTracking PROPERTIES POSITION_INDEPENDENT_CODE ON)
#set_target_properties(CameraTracking PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(${PROJECT_NAME} ${SOURCES})

#Link all obj files
target_link_libraries(${PROJECT_NAME}  ${LIBS}  CameraTracking )  #SDL2::SDL2

#Now copy shaders to build directory

add_custom_target(copy-runtime-files ALL
    COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/shaders ${CMAKE_BINARY_DIR}/shaders
    COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/assets
    DEPENDS ${MY_TARGET})
