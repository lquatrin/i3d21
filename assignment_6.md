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

### Experimenting with Other Shapes


### References

[1] PyTorch3D Loss functions for meshes and point clouds. Available at: https://pytorch3d.readthedocs.io/en/latest/modules/loss.html.
