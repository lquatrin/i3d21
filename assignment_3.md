# 3D Graphics Systems Course - IMPA 2021

#### Leonardo Quatrin Campagnolo

---------

## Assignment 3 - Camera - Bundle Adjustment

In this assignment, the objective was to estimate the extrinsic parameters of a set (bundle) of cameras, given multiple pairs of relative transformations between them. The PyTorch3D API is used to set up an optimization process to minimize the discrepancies between pairs of relative cameras.

### Setup

The problem consists on finding the relative positions between a set of **N** cameras. To achieve this, the rotation and translation matrices between each pair of cameras are estimated. The intrinsic parameters of the cameras are implicitly known, so only the extrinsic parameters are considered. If we consider the epipolar geometry, we are actually computing the essential matrix, which maps one camera to another. 
 
To be able to find a valid solution to this problem, the first camera defines the reference coordinate system. Thus, the solution to the problem consists of finding the relative transformation between each pair of cameras. With that in mind, the first camera is considered the trivial case, where its rotations matrix is the zero vector for translation.

Using this initial state for the first camera, a solution for the bundle adjustment problem can be found by minimizing the discrepancy between each pair of cameras, taking the first camera as our reference coordinate system. We can also visualize this as a normalization process, which can be done for any set of cameras that as given to our optimization process.

### Representing rotations

The initial solution to this problem uses axis-angle reprensetations to store the rotation between each pair of cameras.

### Results

Uma coisa que fiz nesse caso foi rodar a função de otimização sem aplicar a máscara, e vi que ao final as câmeras respeitaram a distância, mas justamente em posições diferentes, como se estivessem em um outro sistema de coordenadas.
