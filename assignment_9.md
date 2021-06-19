# 3D Graphics Systems Course - IMPA 2021

#### Leonardo Quatrin Campagnolo

---------

## Assignment 9 - View Optimization / Fit Mesh

In this assignment, the idea is to practice operations with differentiable rendering using the PyTorch3D to optimize meshes and scene parameters from multiple viewpoints. The code of this assignment can be found [here](https://github.com/lquatrin/i3d21/blob/main/code/a9/Assignment9.ipynb).

### Rendering pipeline and discontinuity issues

A rendering pipeline is used to generate 2d images from 3d scenes, where the scene is defined by 3d objects represented by vertices, lines and polygons. It projects the current scene onto the image plane defined by a camera in a 3D environment. To do that, the camera first transforms each 3D object from world to view coordinates, which can be seem as a change of basis, considering the camera positioned at (0,0,0) and directed along the z direction (being positive or negative depending on the convention). Then, a projection matrix (also given by the camera) projects the triangles onto the image plane, which is considered the xy-plane. A camera also defines which objects will be shown, and which objects will be discarded, when they do not influence the rendered image. In the last step, each pixel is colored by considering the nearest face. Texture mapping and lighting effects are also computed in this stage for each pixel.

According to [1], there are two steps that are not differentiable in rendering. First, the z discontinuity, which happens when two triangles are close in terms of depth, then a small change in its vertices may cause them to overlap. The overlap causes a discontinuity since it causes a step change in pixel color when the nearest face changes. There is also a second discontinuty problem caused by changing the position of a triangle in screen space. In this case, a step change in pixel color is caused due to face boundaries.

In PyTorch3D, they used a soft rasterizer [2] to solve these problems. The problem of z discontinuity is solved by soft aggregating the K nearest faces. The second discontinuity problem was solved by decaying each face’s influence toward its boundary. It is important to notice that PyTorch3D is implemented based on each face that intersects the camera's image plane, instead of considering each pixel, since the face blending have to be calculated.

### Dataset Creation

We sample different camera positions that encode multiple viewpoints of the cow. We create a renderer with a shader that performs texture map interpolation. We render a synthetic dataset of images of the textured cow mesh from multiple viewpoints.


2.1 Present a high level description of a rendering pipeline based on rasterization (not ray tracing!). Which steps are inherently not differentiable? How could we re-design these operations to build a fully differentiable pipeline?

2.2 Place a point light in the scene and render the meshes again using the silhouette renderer. Does it make any difference? Why?

No, because silhouette renderer does not compute lighting, only the top K faces for each pixel, based on the 2d euclidean distance of the center of the pixel to the mesh face. https://pytorch3d.readthedocs.io/en/latest/modules/renderer/shader.html

### Mesh prediction via silhouette rendering

In the previous section, we created a dataset of images of multiple viewpoints of a cow.

Later, we will fit a mesh to the rendered RGB images, as well as to just images of just the cow silhouette. For the latter case, we will render a dataset of silhouette images. Most shaders in PyTorch3D will output an alpha channel along with the RGB image as a 4th channel in an RGBA image. The alpha channel encodes the probability that each pixel belongs to the foreground of the object. We contruct a soft silhouette shader to render this alpha channel.

====

In this section, we predict a mesh by observing those target images without any knowledge of the ground truth cow mesh. We assume we know the position of the cameras and lighting.

We first define some helper functions to visualize the results of our mesh prediction:

====

Starting from a sphere mesh, we will learn offsets of each vertex such that the predicted mesh silhouette is more similar to the target silhouette image at each optimization step. We begin by loading our initial sphere mesh:

====
We initialize settings, losses, and the optimizer that will be used to iteratively fit our mesh to the target silhouettes:

===

We write an optimization loop to iteratively refine our predicted mesh from the sphere mesh into a mesh that matches the sillhouettes of the target images:

3.1 Visualize the deformed mesh using Plotly and describe it qualitatively in comparinson with the target mesh. You can also download it and visualize it in another software if you wish.

3.2 Experiment changing the number of images num_views_per_iteration used to compute the silhouette loss each iteration. Would it still work if we computed the loss using num_views_per_iteration? What if we only had a single image, a single point of view, in our dataset?

3.3 Compare the target and source meshes sizes (number of vertices and faces). Are they close? Does the final result improve if you start from a source mesh with more vertices?


#### Using higher level icosphere 

```python
Cow Mesh Vertices: 2930
Sphere level 4 Mesh Vertices: 2562
Sphere level 5 Mesh Vertices: 10242
```

### Mesh and texture prediction via textured rendering

We can predict both the mesh and its texture if we add an additional loss based on the comparing a predicted rendered RGB image to the target image. As before, we start with a sphere mesh. We learn both translational offsets and RGB texture colors for each vertex in the sphere mesh. Since our loss is based on rendered RGB pixel values instead of just the silhouette, we use a SoftPhongShader instead of a SoftSilhouetteShader.


We initialize settings, losses, and the optimizer that will be used to iteratively fit our mesh to the target RGB images:


We write an optimization loop to iteratively refine our predicted mesh and its vertex colors from the sphere mesh into a mesh that matches the target images:


4.1 Compare the target and source meshes and describe the result qualitatively.

4.2 Do you think it could be better? Analyze the losses values, the meshes, the hyperparemeters and try other values. Even if you don't get a better result, try to explain your intutition for the changes you made.

4.3 Make a copy of the target mesh and now try to optimize the texture only, starting from the ground truth geometry. Describe your result.

[EXTRA] E.1 Deform the textured cow into a sphere - you can choose to supervise with multi-view images (of a sphere) or the sphere geometry. Save the result, then deform the textured sphere into the cow again. What happened?




### Camera Position Optimization

Until now, we assumed we knew the cameras and we learned how to infer the geometry and texture of a mesh using a differentiable renderer and supervising the training with the multiview images dataset.

Now, we'll assume we have the geometry and images of the object, but we don't know the cameras. Can we infer the camera position by backpropagation?

Here we create a simple model class and initialize a parameter for the camera position.

Initialize the model and optimizer
Now we can create an instance of the model above and set up an optimizer for the camera position parameter.

Run the optimization
We run several iterations of the forward and backward pass and save outputs every 10 iterations. When this has finished take a look at ./cow_optimization_demo.gif for a cool gif of the optimization process!

5.1 Evaluate how close our prediction is to the ground truth camera position. Explain your metric.

5.2 Experiment with other views - different target images and different initial positions to the camera model. Does it always work?

EXTRA E.2: Could you estimate the scene illumination - in this case, the location of a single point light? Set up and run an example

### Light Position Optimization

The last experiment of this assignment was to optimize a light position given the camera 


### References

[1] Accelerating 3D Deep Learning with PyTorch3D

[2] Soft Rasterizer: Differentiable Rendering for Unsupervised Single-View Mesh Reconstruction