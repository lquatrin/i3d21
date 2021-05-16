# 3D Graphics Systems Course - IMPA 2021

#### Leonardo Quatrin Campagnolo

---------

## Assignment 6 - Deform a source mesh to form a target mesh using 3D loss functions

In this assignment, the objective was learn how to deform an initial shape to fit a target shape. We start by using a sphere to deform into a target mesh, using an optimization procedure to offset each vertex at each step, making it closer to the target mesh. To be able to do that, we use different PyTorch3D mesh loss functions and evaluate the optimized results.

We use the MeshLab system to better visualize the generated meshes.

### Creating the mesh object

To speed up the optimization process, we must ensure that each mesh is normalized. So, we center the mesh at the origin by translating all vertices using its mean position. We also scale all vertices to stay at [-1, 1].

### Deforming a Sphere mesh to a Dolphin


### Experimenting with Other Shapes
