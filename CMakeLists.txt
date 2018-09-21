#====================
cmake_minimum_required (VERSION 3.9)
PROJECT(ICP_Demo LANGUAGES CXX CUDA)

# Useful CMake options
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_FLAGS "-std=c++11 -fpermissive")
set(OpenGL_GL_PREFERENCE "GLVND")

# Search desired Qt packages
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)  #only need this till nvcc starts supporting c++17 compiler

set(INCLUDE_DIRS  ${INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS} ${SDL2_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OPENGL_LIBRARY} ${CUDA_LIBRARIES} ${SDL2_LIBRARIES} ${GLEW_LIBRARIES} ${Boost_LIBRARIES})

if(CUDA_FOUND)
  # compared to class settings, we let NVidia's FindCUDA CMake detect 
  # whether to build x64.  We tell it to support most devices, though, 
  # to make sure more people can easily run class code without knowing 
  # about this compiler argument
  set(CUDA_NVCC_FLAGS "
  -ccbin /usr/bin/g++-5; 
  -gencode;arch=compute_30,code=sm_30;  
  ")

  # add -Wextra compiler flag for gcc compilations
  if (UNIX)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
  endif (UNIX)

  # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-g -G")

else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()

#specify include directory
#include_directories(/usr/include)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(INCLUDE_DIRS ${INCLUDE_DIRS})

set(SOURCES DepthMain.cpp camera.cpp Window.cpp Application.cpp prereq.h ShaderProgram.hpp Frustum.cpp camera.h Window.h
    Application.h Frustum.h cudaHelper.h CameraTracking.h)

#Compile
add_library(CameraTracking SHARED CameraTracking.cu )
#set_target_properties(CameraTracking PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(CameraTracking PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(${PROJECT_NAME} ${SOURCES})

#Link all obj files
target_link_libraries(${PROJECT_NAME}  ${LIBS} SDL2::SDL2 CameraTracking )

#Now copy shaders to build directory

add_custom_target(copy-runtime-files ALL
    COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/shaders ${CMAKE_BINARY_DIR}/shaders
    COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/assets
    DEPENDS ${MY_TARGET})
