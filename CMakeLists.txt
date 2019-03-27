#====================
cmake_minimum_required (VERSION 3.9)
PROJECT(ICP_Demo LANGUAGES C CXX CUDA)

# Useful CMake options
#set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.2/bin/nvcc")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_FLAGS "-std=c++11 -fpermissive")
set(OpenGL_GL_PREFERENCE "GLVND")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_30,code=sm_30")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) #ensures CMake creates compile_commands.json
#set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

IF( EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json" )
  EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
  )
ENDIF()

find_package(CUDA REQUIRED)
#find_package(Armadillo REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)
find_package(glm REQUIRED)
message(STATUS "GLM included at ${GLM_INCLUDE_DIR}")
find_package(Boost COMPONENTS filesystem REQUIRED)  #only need this till nvcc starts supporting c++17 compiler

set(CUHELPER_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/cuda_helper")

#Compile GladLib
set(GLAD_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/glad/include")
set(GLAD_SRC "${GLAD_INCLUDE_DIR}/../src/glad.c")
add_library("GladLib" ${GLAD_SRC})
target_include_directories("GladLib" PRIVATE "${GLAD_INCLUDE_DIR}")

#Compile CameraTrackingUtils
#add_library(CameraTrackingUtils STATIC LinearSystem.cu CameraTrackingUtils.cu)
add_library(CameraTrackingUtils STATIC Solver.cu CameraTrackingUtils.cu)
set_target_properties(CameraTrackingUtils PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(CameraTrackingUtils PROPERTIES POSITION_INDEPENDENT_CODE ON)
#set_target_properties(CameraTrackingUtils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(INCLUDE_DIRS  ${INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${CUHELPER_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${GLM_INCLUDE_DIR}
                ${OPENGL_INCLUDE_DIRS} ${SDL2_INCLUDE_DIRS} ${GLAD_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})  #${ARMADILLO_INCLUDE_DIR}

set(LIBS ${LIBS} ${OPENGL_LIBRARY} ${CUDA_LIBRARIES}  ${SDL2_LIBRARIES} ${Boost_LIBRARIES} ${GladLib})  #${ARMADILLO_LIBRARY}

#specify include directory
#include_directories(${CMAKE_SOURCE_DIR} )
include_directories(${INCLUDE_DIRS})

set(SOURCES DepthMain.cpp camera.cpp Window.cpp Application.cpp Solver.cpp CameraTracking.cpp prereq.h ShaderProgram.hpp Frustum.cpp camera.h Window.h
  Application.h Frustum.h CameraTracking.h Solver.h EigenUtil.h termcolor.hpp DebugHelper.hpp ${GLAD_SRC})


add_executable(${PROJECT_NAME} ${SOURCES})

#Link all obj files
target_link_libraries(${PROJECT_NAME}   CameraTrackingUtils ${LIBS} SDL2 -lcublas)  #SDL2::SDL2

#Now copy shaders to build directory

add_custom_target(copy-runtime-files ALL
    COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/shaders ${CMAKE_BINARY_DIR}/shaders
    COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/assets
    DEPENDS ${MY_TARGET})
