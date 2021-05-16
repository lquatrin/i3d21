# 3D Graphics Systems Course - IMPA 2021

#### Leonardo Quatrin Campagnolo

---------

## Assignment 6 - Deform a source mesh to form a target mesh using 3D loss functions

In this assignment, the objective was learn how to deform an initial shape to fit a target shape. We start by using a sphere to deform into a target mesh, using an optimization procedure to offset each vertex at each step, making it closer to the target mesh. To be able to do that, we use different PyTorch3D mesh loss functions and evaluate the optimized results.

We use the MeshLab system to better visualize the generated meshes.

### Creating the mesh object

To speed up the optimization process, we must ensure that each mesh is normalized. So, we center the mesh at the origin by translating all vertices using its mean position. We also scale all vertices to stay at [-1, 1]. Then we have a normalized sphere as our source mesh, and a normalized dolphin as our target mesh:

![Source and target Mesh](imgs/a6/source_target.png)

### Deforming a Sphere mesh to a Dolphin

The optimization procedure wants to learn how to offset the vertices of the source mesh to make it closer to the target mesh. To achieve this, the distance between the predicted and the target meshes are computed. The chamfer distance is used by sampling a point cloud from each mesh at each step of the optimization loop. In addition, other 3 loss functions are added into the loss function to ensure smoothness of the predicted mesh: **mesh_edge_length**, **mesh_normal_consistency**, and **mesh_laplacian_smoothing** [1].

After 5000 iterations, we have the following result:

![Predicted Dolphin](imgs/a6/p_dolphin_point_cloud.png) ![Target Dolphin](imgs/a6/dolphin_point_cloud.png)

Using different weights for each loss function:

```python
w_chamfer   = 1.00
w_edge      = 1.00
w_normal    = 0.01
w_laplacian = 0.10
``` 

And here we have the sphere deformed into a dolphin given the result from the optimization procedure:

![Predicted and target Mesh](imgs/a6/c_dolphin.png)

Using MeshLab, we can compare the resulting meshes:

<img src="imgs/a6/p_dolphin_meshlab.png" width="30%"><img src="imgs/a6/dolphin_meshlab.png" width="30%"><img src="imgs/a6/c_dolphin_meshlab.png" width="30%">

We notice how the fins, tail and rostrum have been smoothed by the optimization procedure, resulting in a mesh with some differences compared with the target. We can also note how the curvatures in general have also been smoothed by the optimization procedure.

Taking a look at each loss function captured during the optimization loop, we see how chamfer distance decreases at the first iterations, then the normal loss decreasing more slowly.

![Loss Functions](imgs/a6/p_dolphin_losses.png)

Then, we made other two tests checking if we're able to reach a good result with lower iterations. First we made a test with 2500 iterations, resulting in the following mesh:

<img src="imgs/a6/predicted_mesh_2500.png" width="40%">

We saw how some triangles are not well defined, due to the normal loss. We then tested with 3000 iterations:

<img src="imgs/a6/predicted_mesh_3000.png" width="40%">

We then achieved a reasonable result even using 2000 less iterations than the initial experiment.

The current results were achieved considering the linear combination of four losses. If we consider only the camfer distance, we approximate the point cloud of both meshes:

![Chamfer predicter point cloud](imgs/a6/chamfer_predicter_dolphin.png)

However, we're not able to keep the mesh integrity:

<img src="imgs/a6/chamfer_dolphin_mesh00.png" width="40%">

Changing the parameters of each loss function may not result in a solution. Using a higher value for **mesh_normal_consistency** may deform to much the mesh for each optimization loop, ending in a result far from the target:

![Predicted mesh with 0.5 normal consistency](imgs/a6/e_1_dolphin_normal.png)

The above image was generated using the following weights:

```python
w_chamfer   = 1.00
w_edge      = 1.00
w_normal    = 0.50
w_laplacian = 0.10
``` 

We also experiment to decrease the weight of the **mesh_edge_loss**, to prevent a high edge length regularization. In this case, we wan to check if using a lower weigth in this loss function can better approximate the extremities of the dolphin. We compare the result with the standard weigths and with the target mesh:

<img src="imgs/a6/e_1_predicted.png" width="30%"><img src="imgs/a6/e_1_standard.png" width="30%"><img src="imgs/a6/e_1_target.png" width="30%">

We can see the rostrum slightly better in this result. The above image was generated using the following weights:

```python
w_chamfer   = 1.00
w_edge      = 0.10
w_normal    = 0.01
w_laplacian = 0.10
``` 

All the experiments were generated using the SGD optimizer. We also tested the Adam and RMSprop to apply deformation to the mesh. The Adam optimizer required more steps to converge.

TODO RMSProp

### Experimenting with Other Shapes

Now we want to deform a mesh into a mug:

![Mug](imgs/a6/mug.png)

In this case, we won't be able to use a sphere as our source mesh, because we must have a homeomorphism between the topological spaces of both meshes. It means that, from a topological viewpoint, they are the same space and share the same topological properties. It results in a continuous mapping with a continuous inverse function that we want to find via an optimization process to correctly deform our source mesh to the target mesh.

According to the Theorem of closed surfaces, a closed surface is homeomorphic to a sphere or a N-tori, considering its  corresponding genus N, which defines the number of holes of a closed surface. It this case, the sphere has genus 0 and the mug has genus 1, so we cannot find a continuous mapping between these meshes.

Indeed, if we try to deform the sphere into the mug, we get the following result:

<img src="imgs/a6/mug_predicted_sphere.png" width="30%"><img src="imgs/a6/mug_target.png" width="30%"><img src="imgs/a6/mug_predicted_sphere_c.png" width="30%">

Also computing the losses per iteration:

![Loss Functions](imgs/a6/p_mug_losses.png)

We also tried to deform using a torus as our source mesh, available from PyTorch3D:

![Torus and Mug](imgs/a6/torus_and_mug.png)

In theory, we should be able to deform the torus into the mug, however, we achieved the following result:

<img src="imgs/a6/mug_predicted_torus.png" width="30%"><img src="imgs/a6/mug_target_torus.png" width="30%"><img src="imgs/a6/mug_predicted_torus_c.png" width="30%">

With losses per iteration:

![Loss Functions](imgs/a6/p_mug_losses_2.png)

Since the chamfer distance is used to approximate the target mesh, we think that there is a possibility of getting the wrong samples to compute the distance, which cause the mesh to diverge from what we expect. We tried a last experiment using only the chamfer distance to check the resulting predicted mesh only considering a point cloud.

ADD IMAGE

With losses per iteration:

![Loss Functions](imgs/a6/p_mug_losses_3.png)

We also made an additional test with a genus 0 mesh:

![Among Us](imgs/a6/among_us_mesh.png)

Since it have genus 0, we can approximate using a normalized sphere. The predicted mesh is:

<img src="imgs/a6/among_predicted.png" width="30%"><img src="imgs/a6/among_target.png" width="30%"><img src="imgs/a6/among_c.png" width="30%">

### References

[1] PyTorch3D Loss functions for meshes and point clouds. Available at: https://pytorch3d.readthedocs.io/en/latest/modules/loss.html.

[2] Introduction to Topology: Classification of Surfaces. Available at: https://people.math.osu.edu/fiedorowicz.1/math655/classification.html.
