#====================
cmake_minimum_required (VERSION 2.8)
PROJECT(ICP_Demo)

# Useful CMake options for Qt projects
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_FLAGS "-std=c++0x -g")
set (CMAKE_CXX_FLAGS "-fpermissive -fPIC")
set(OpenGL_GL_PREFERENCE "GLVND")

# Search desired Qt packages
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)

set(INCLUDE_DIRS  ${INCLUDE_DIRS} ${CUDA_INCLUDE_DIRECTORIES} ${OPENGL_INCLUDE_DIRS} ${SDL2_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OPENGL_LIBRARY} ${SDL2_LIBRARIES} ${GLEW_LIBRARIES} ${Boost_LIBRARIES})

if(CUDA_FOUND)
  # compared to class settings, we let NVidia's FindCUDA CMake detect 
  # whether to build x64.  We tell it to support most devices, though, 
  # to make sure more people can easily run class code without knowing 
  # about this compiler argument
  set(CUDA_NVCC_FLAGS "
  -ccbin /usr/bin/g++-5; 
  -gencode;arch=compute_30,code=sm_30;  
  -gencode;arch=compute_35,code=sm_35;
  -gencode;arch=compute_35,code=compute_35;
  ")

  # add -Wextra compiler flag for gcc compilations
  if (UNIX)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
  endif (UNIX)

  # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

  CUDA_ADD_EXECUTABLE(testCUDA testCUDA.cu)
else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()

#specify include directory
#include_directories(/usr/include)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(INCLUDE_DIRS ${INCLUDE_DIRS} )

set(SOURCES DepthMain.cpp camera.cpp Window.cpp Application.cpp prereq.h ShaderProgram.hpp Frustum.cpp camera.h Window.h
    Application.h Frustum.h)

#Compile
add_executable(${PROJECT_NAME} ${SOURCES})

#Link all obj files
target_link_libraries(${PROJECT_NAME} ${LIBS} SDL2::SDL2)

#Now copy shaders to build directory

add_custom_target(copy-runtime-files ALL
    COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/shaders ${CMAKE_BINARY_DIR}/shaders
    COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/assets
    DEPENDS ${MY_TARGET})
