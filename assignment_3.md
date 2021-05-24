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

The initial solution to this problem uses the log axis-angle representation to store the rotation between each pair of cameras. Using PyTorch3D API, we can recover the 3x3 rotation matrix from the axis-angle using the method **so3_exponential_map**. There are a few possible solutions to how we can represent each  rotation matrix:

1. Fixed/Euler angles: In this representation, the rotations are computed as a sequence of rotations, using the axes of a coordinate system (global or local). The final rotation is computed by accumulating all rotations around each axis (Rx, Ry, Rz). 

There are a few problems when using this reprensetation: first, they are not commutative, i.e., the order matters. So, if we use different sequences, it will generate different final rotations. Also, they present a singularity problem, which cause the loss of one degree of freedom in three-dimensions, known as "gimbal lock". It is also difficult to provide a smooth interpolation between two distinct points. Finally, a slightly change in the parameters does not represent the same rotation of the rigid body. We can take the rotation around the pole of a sphere as an example: even rotating the point around the sphere, the distance does not correspond at the same amount if we rotate the same point at its equator.

2. Axis angle: Used as the solution to this assignment. In this case, the rotation is calculated based on an arbitrary axis, which is defined by the composition of 3 angles. According to PyTorch3D documentation, the arbitrary axis is calculated using the Rodriguez Formula, from method **so3_exponential_map**.

This representation does not generate the singularity problem, but we must convert back to a matrix to be able to composite a rotation. In addition, there is an ambiguity if we have the same axis with exchanged sign, since they can represent the same rotation. Still, they present the same limitation from the euler angles: it is difficult to provide a smooth interpolation between two points, and the small change of the parameters does not represent the same amount in a rigid body.

3. Quaternion: Considered the most interesting way to represent rotations, which can also provide a smooth interpolation, and consistent rotation between the parameters and the model. However, it requires a 4D vector to represent each rotation. Similar to Euler angles, the order of each rotation matters, which can lead to different results. Also, there is a particular property of quaternions that, if we want to generate a set of random quaternions, it will be more uniform at a sphere.

4. A recent paper used SVD to predict rotations, which lead to better results for different applications. In this case, 9 values are used to represent rotations[1].

### Results

Uma coisa que fiz nesse caso foi rodar a função de otimização sem aplicar a máscara, e vi que ao final as câmeras respeitaram a distância, mas justamente em posições diferentes, como se estivessem em um outro sistema de coordenadas.

For the results section, we start by showing the result using the optimization procedure using the axis-angle representation. After 2000 iterations, we're able to achieve the following result:




With the loss vs iteration graph below:



To improve the approximation, i tried to not randomly initialize the rotation and translation of all cameras, but making all start at the trivial location. It makes sense that a better initial guess may lead to a better approximation after the optimization procedure.











### References

[1] An Analysis of SVD for Deep Rotation Estimation.

