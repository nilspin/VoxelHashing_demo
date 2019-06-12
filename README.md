# ICP Demo

This a simple point-cloud registration program using Point-to-plane Iterative Closest Points algorithm.
To build, you'll need the following (version in brackets are versions used during development)  :

* cmake (>3)
* gcc (7.4) - because anything newer doesn't play with CUDA
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
* Build for windows
* TSDF fusion
