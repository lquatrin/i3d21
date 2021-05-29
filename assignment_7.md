# 3D Graphics Systems Course - IMPA 2021

#### Leonardo Quatrin Campagnolo

---------

## Assignment 7 - Render Point Clouds and Meshes

In this assignment, the idea is to practice operations with 3D meshes and point clouds using the PyTorch3D.

### Loading a textured mesh and visualizing its texture map

In this report, the cow mesh was used for the experiments, which can be loaded from PyTorch3D repository. The function **load_objs_as_meshes** can be used to load all the data from the current object. It contains a UV coordinate defined for each vertex, and we can visualize the current UV mapping using the function **texturesuv_image_matplotlib** from pytorch3d.vis.texture_vis:

![Texture Map](imgs/a7/cow_mesh_texture_map.png)

where the red dots defines the UV coordinate of each vertex.

### Rendering a Mesh

First, the **MeshRasterizer** is used to render the mesh, using a **SoftPhongShader**:

![Cow mesh with SoftPhongShader](imgs/a7/2_0.png)

It is possible to change some of the parameters:

### Moving the scene


### Batched Rendering

Creating a batch of different cameras it is possible to visualize the mesh with different viewpoints:

![Batch of cows](imgs/a7/3_0.png)

### Rendering Point Clouds


### Plotly visualization of Point Clouds
