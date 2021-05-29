# 3D Graphics Systems Course - IMPA 2021

#### Leonardo Quatrin Campagnolo

---------

## Assignment 3 - Camera - Bundle Adjustment

In this assignment, the objective was to estimate the extrinsic parameters of a set (bundle) of cameras, given multiple pairs of relative transformations between them. The PyTorch3D API is used to set up an optimization process to minimize the discrepancies between pairs of relative cameras.

### Setup

The problem consists on finding the relative positions between a set of **N** cameras. To achieve this, the rotation and translation matrices between each pair of cameras are estimated. The intrinsic parameters of the cameras are implicitly known, so only the extrinsic parameters are considered. If we consider the epipolar geometry, we are actually computing the essential matrix, which maps one camera to another. 
 
To be able to find a valid solution to this problem, the first camera defines the reference coordinate system. Thus, the solution to the problem consists of finding the relative transformations between each pair of cameras. With that in mind, the first camera is considered the trivial case, where its rotations matrix is the zero vector for translation.

Using this initial state for the first camera, a solution for the bundle adjustment problem can be found by minimizing the discrepancy between each pair of cameras, taking the first camera as our reference coordinate system. We can also visualize this as a normalization process, which can be done for any set of cameras that as given to our optimization process.

### Representing rotations

The initial solution to this problem uses the log axis-angle representation to store the rotation between each pair of cameras. Using PyTorch3D API, we can recover the 3x3 rotation matrix from the axis-angle using the method **so3_exponential_map**. There are a few possible solutions to how we can represent each rotation matrix:

#### 1. Fixed/Euler angles:

In this representation, the rotations are computed as a sequence of rotations, using the axes of a coordinate system (global or local). The final rotation is computed by accumulating all rotations around each axis (Rx, Ry, Rz). 

There are a few problems when using this reprensetation: first, they are not commutative, i.e., the order matters. So, if we use different sequences, it will generate different final rotations. Also, they present a singularity problem, which cause the loss of one degree of freedom in three-dimensions, known as "gimbal lock". It is also difficult to provide a smooth interpolation between two distinct points. Finally, a slightly change in the parameters does not represent the same rotation of the rigid body. We can take the rotation around the pole of a sphere as an example: even rotating the point around the sphere, the distance does not correspond at the same amount if we rotate the same point at its equator.

#### 2. Axis angle:

Used as the solution to this assignment. In this case, the rotation is calculated based on an arbitrary axis, which is defined by the composition of 3 angles. According to PyTorch3D documentation, the arbitrary axis is calculated using the Rodriguez Formula, from method **so3_exponential_map**.

This representation does not generate the singularity problem, but we must convert back to a matrix to be able to composite a rotation. In addition, there is an ambiguity if we have the same axis with exchanged sign, since they can represent the same rotation. Still, they present the same limitation from the euler angles: it is difficult to provide a smooth interpolation between two points, and the small change of the parameters does not represent the same amount in a rigid body.

#### 3. Quaternion:

Considered the most interesting way to represent rotations, which can also provide a smooth interpolation, and consistent rotation between the parameters and the model. However, it requires a 4D vector to represent each rotation. Similar to Euler angles, the order of each rotation matters, which can lead to different results. Also, there is a particular property of quaternions that, if we want to generate a set of random quaternions, it will be more uniform at a sphere.

#### 4. Other representations:

A recent paper used SVD to predict rotations, which lead to better results for different applications. In this case, 9 values are used to represent rotations [1]. In this report, i didn't try SVD representation for the bundle adjustment problem.

### Optimization Results

The SGD optimizer was used to optimize both rotations and translations of each camera. For each loop step, the essential matrix between each pair of cameras is evaluated and compared with a ground truth, which defines the loss function **camera_distance**. We initialize the first camera as the trivial case, and the others are initialized with random values. In the first experiment, the axis angle representation is used to compute the rotation matrix of each camera.

For the first result, using 2000 iterations, we reached a result with camera_distance = 4.597e-03 at the last iteration:

![Bundle after optimization](imgs/a3/camera_std.png)

with the cameras in purple being our ground truth, and the orange cameras being the approximated ones. The graph of loss vs iterations shows how the distance between the cameras are decreasing during the optimization loop. We can see that in the first steps, the accumulated loos is higher, since we start the cameras at random states:

![Loss per iteration](imgs/a3/loss.png)

Using the Cow Mesh, we can make a qualitative comparison between the optimized cameras with our ground truth:

Initial state of the cameras:

![Camera init](imgs/a3/init.png)

Images generated with the Ground Truth cameras:

![Camera ground truth](imgs/a3/gt.png)

Images generated with the optimized cameras:

![Camera optimized](imgs/a3/approx.png)

We can note that the first camera does not change during the optimization process, since we define it as the trivial case. We can also note small differences between the ground truth and the optimized cameras. Take the image from the second row and third column as an example: we can note how it generated a slightly different image.

### Additional Results

To improve the approximation, i tried to not randomly initialize the rotation and translation of the cameras, but making all start as the trivial case. It seemed to be a better initial guess instead of just using random values. In this case, running the same optimization loop, i was able to achieve a result of camera_distance = 7.092e-08 at the last iteration:

![Cameras with trivial initialization](imgs/a3/camera_init.png)

We can see by the images generated after this optmization are closer to the ground truth:

![Images wih trivial initialized_cameras](imgs/a3/images_init.png)

Then, i made an additional experiment representing each rotation as a quaternion. It also seemed to be a good representation since its interpolation works better than using axis-angle for rotations. However, the quaternions were transformed back to a matrix (using the method **quaternion_to_matrix**) to compute the relative cameras and the loss function. We also apply the trivial initialization for all cameras, but in this case, we initialize each rotation with the identity quaternion. I was able to reach a camera_distance = 4.957e-08 as the final loss:

![Cameras with quaternions](imgs/a3/camera_quat.png)

We can see the gerenated images using the quaternion representation are also similar to the ground truth, since we achieved a good result:

![Images wih quaternions](imgs/a3/images_quat.png)

In this case, i also think some modifications could be done in the optimization loop to compare quaternions properly, but I ended up not progressing in this experiment. Finally, we compute the loss function for each experiment:

![Computed Losses](imgs/a3/all_losses.png)

### References

[1] Levinson, J., Esteves, C., Chen, K., Snavely, N., Kanazawa, A., Rostamizadeh, A., & Makadia, A. (2020). An Analysis of SVD for Deep Rotation Estimation. ArXiv, abs/2006.14616.



