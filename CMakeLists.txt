#====================
cmake_minimum_required (VERSION 3.9)
PROJECT(VoxelHashing LANGUAGES C CXX CUDA)

# Useful CMake options
#set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.2/bin/nvcc")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_FLAGS "-std=c++11 -fpermissive -w")
set(OpenGL_GL_PREFERENCE "GLVND")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_61,code=sm_61 -w") #new laptop
#string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_30,code=sm_30 -w")  #old, main laptop
#if( CMAKE_BUILD_TYPE MATCHES "Debug" )
string( APPEND CMAKE_CUDA_FLAGS " -g -G" )  #TODO : Remove these 3 lines later
string( APPEND CMAKE_CUDA_FLAGS " -lineinfo" )
string( APPEND CMAKE_CUDA_FLAGS " -fmad=false" )
#endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) #ensures CMake creates compile_commands.json
#set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

IF( EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json" )
  EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
  )
ENDIF()

find_package(CUDA REQUIRED)
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
add_library(GladLib ${GLAD_SRC})
target_include_directories(GladLib PRIVATE "${GLAD_INCLUDE_DIR}")

add_library(FBO STATIC FBO.hpp)
set_target_properties(FBO PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(FBO GladLib)

add_library(SDFRenderer STATIC SDFRenderer.h SDFRenderer.cpp)
set_target_properties(SDFRenderer PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(SDFRenderer GladLib)

add_library(ShaderProgram STATIC ShaderProgram.hpp)
set_target_properties(ShaderProgram PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(ShaderProgram GladLib)


add_library(Window STATIC Window.cpp)
target_link_libraries(Window ${SDL2_LIBRARIES})

add_library(Camera STATIC camera.cpp)
target_link_libraries(Camera ${SDL2_LIBRARIES})

add_library(Frustum STATIC Frustum.cpp)
#target_link_libraries(Frustum ShaderProgram)

#Compile CameraTrackingUtils
add_library(Solver STATIC SE3.h SE3.cpp Solver.cpp Solver.cu)
set_target_properties(Solver PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(Solver PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Solver ${CUDA_CUBLAS_LIBRARIES})

add_library(CameraTrackingUtils STATIC CameraTracking.cpp CameraTrackingUtils.cu)  #SE3.hpp
set_target_properties(CameraTrackingUtils PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(CameraTrackingUtils PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(CameraTrackingUtils Solver)
#set_target_properties(CameraTrackingUtils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#Compile IntegrationUtils
add_library(IntegrationUtils STATIC VoxelDataStructures.h VoxelUtils.h VoxelUtils.cu)	#VoxelDataStructures.h VoxelUtils.h 
set_target_properties(IntegrationUtils PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(IntegrationUtils PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(SDFModule STATIC VoxelUtils.h SDF_Hashtable.h SDF_Hashtable.cpp)	# VoxelUtils.h SDF_Hashtable.h
target_link_libraries(SDFModule IntegrationUtils)


add_library(Application STATIC Application.h Application.cpp)
target_link_libraries(Application GladLib)
target_link_libraries(Application Camera)
target_link_libraries(Application Window)
target_link_libraries(Application SDFModule)
target_link_libraries(Application SDFRenderer)
target_link_libraries(Application Frustum)
target_link_libraries(Application CameraTrackingUtils)
#target_link_libraries(Application IntegrationUtils)

set(INCLUDE_DIRS  ${INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${CUHELPER_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${GLM_INCLUDE_DIR}
                ${OPENGL_INCLUDE_DIRS} ${SDL2_INCLUDE_DIRS} ${GLAD_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})  
set(LIBS ${LIBS} ${OPENGL_LIBRARY} ${CUDA_LIBRARIES}  ${SDL2_LIBRARIES} ${Boost_LIBRARIES} )  # ${GladLib} ${ShaderProgram} ${Window} ${Camera}

#specify include directory
#include_directories(${CMAKE_SOURCE_DIR} )
include_directories(${INCLUDE_DIRS})

#set(HEADERS prereq.h common.h camera.h Window.h ShaderProgram.hpp Solver.h SE3.hpp CameraTracking.h VoxelUtils.h VoxelDataStructures.h SDF_Hashtable.h Application.h EigenUtil.h termcolor.hpp DebugHelper.hpp)
set(SOURCES DepthMain.cpp )  #Frustum.cpp Frustum.h Application.cpp camera.cpp Window.cpp


add_executable(${PROJECT_NAME} ${SOURCES} ) #${HEADERS}

#Link all obj files
target_link_libraries(${PROJECT_NAME} ${LIBS} Application)  #SDL2::SDL2  CameraTrackingUtils

#Now copy shaders to build directory

add_custom_target(copy-runtime-files ALL
    COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/shaders ${CMAKE_BINARY_DIR}/shaders
    COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/assets
    DEPENDS ${MY_TARGET})
