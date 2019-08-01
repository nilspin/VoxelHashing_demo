# VoxelHashing demo

---
1/8/2019 Update: I am considering stopping development here. Image-tracking, and depth fusion into global model work. The renderer doesn't.  
I've spent too much time trial-and-erroring my way through shaders but without proper glsl debug tooling (or better GPU) I dont think I'll be able to resolve it. I am not abandoning the project, but will continue once I have a job, and better computer.

The latest code lies in 'master' branch. It is not very clean but if you want to build it yourself steps are given below. Let me know if they don't work.  

---

A program for GPU based scalable 3D reconstruction based on Prof. Matthias Niessner's 2013 SIGGRAPH paper [VoxelHashing](http://niessnerlab.org/papers/2013/4hashing/niessner2013hashing.pdf).

It ~~has~~ will have 3 components :  
1. **Image tracking** : Point-cloud registration using Point-to-plane Iterative Closest Points algorithm. Uses Gauss-Newton for solving non-linear least squares problem.  
2. **Storage** : Hash-table in GPU memory that stores underlying 3D model as 'chunks' of voxels, and is backbone of the program. Supports fast addition and deletion of blocks on-the-fly.   
3. **[WIP] Renderer** : For visualising integrated model so far.  

To build, you'll need the following (version in brackets are versions used during development)  :

* CMake (>3.9)
* gcc (7.4) - because anything newer doesn't play well with CUDA
* (Windows) Visual Studio Community 2015 (v14.0)
* CUDA (10.1)
* GLM (0.9.9.2)
* SDL2 (2.0.8)
* Eigen3 with default OpenBLAS (3.3.5)
* Boost-filesystem (1.68.0)

Also, change the SM version in CMakeLists according to your GPU. 

## To build on Windows
* Easiest way to build is using cmake-gui. Point to the correct include/lib paths for SDL2, GLM, Eigen, Boost, CUDA etc and you should be good to go.
* Make sure Boost is compiled with MSVC v140 toolset or just download binaries from [here](https://sourceforge.net/projects/boost/files/boost-binaries/).
* Copy `SDL2.dll` provided from your SDL2 install to build folder.

## To build on Linux:
Change "`${SDL2_LIBRARIES}`" to "`SDL2`" inside CMakeLists.txt. SDL2 doesn't link otherwise.  

    export CC=gcc-7
    export CXX=g++-7
    mkdir build
    cd build
    cmake ..
    make -j
    optirun ./ICP_Demo



## Todo:
* ~~Depth Integration~~
* Write renderer that reads from hashtable
* KinectV2 integration
* Refactor, Optimisation
