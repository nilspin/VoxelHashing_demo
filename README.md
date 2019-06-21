# ICP Demo

A program for GPU based scalable 3D reconstruction based on Matthias Niessner's 2013 SIGGRAPH paper [VoxelHashing](http://niessnerlab.org/papers/2013/4hashing/niessner2013hashing.pdf).

It ~~has~~ will have 3 components :  
1. Point-cloud registration using Point-to-plane Iterative Closest Points algorithm. Uses Gauss-Newton for solving non-linear least squares problem.  
2. [WIP] Hash-table in GPU memory that stores underlying 3D model by-parts, and is backbone of the program. Supports fast addition and deletion of blocks on-the-fly.   
3. [TODO] Renderer for visualising integrated model so far.  

To build, you'll need the following (version in brackets are versions used during development)  :

* cmake (>3)
* gcc (7.4) - because anything newer doesn't play well with CUDA
* (Windows) Visual Studio Community 2015 (v14.0)
* CUDA (10.1)
* GLM (0.9.9.2)
* SDL2 (2.0.8)
* Eigen3 with default OpenBLAS (3.3.5)
* Boost-filesystem (1.68.0)

## To build on Linux:

    export CC=gcc-7
    export CXX=g++-7
    mkdir build
    cd build
    cmake ..
    make -j

## Todo:
* Depth Integration
* Write renderer that reads from hashtable