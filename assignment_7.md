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

In PyTorch3D, a renderer is composed of a rasterizer and a shader which each have a number of subcomponents such as a camera (orthographic/perspective). First, the **MeshRenderer** will be used with a perspective camera and a point light, using the  **MeshRasterizer** and the **SoftPhongShader**:

![Cow mesh with SoftPhongShader](imgs/a7/2_0.png)

It is possible to update the rendering parameters by passing it when generating a new batch of images, such as light source position or material.

![Cow mesh with different light position](imgs/a7/2_1.png)

![Cow mesh with different material](imgs/a7/2_3.png)


Also changing the shader, it is possible to generate different effects. Here we have another result of the cow mesh using a new **MeshRenderer** with **HardFlatShader**:

![Cow mesh with hardflatshader](imgs/a7/2_2.png)

### Moving the scene


### Batched Rendering

Creating a batch of different cameras it is possible to visualize the mesh with different viewpoints:

![Batch of cows](imgs/a7/3_0.png)

### Rendering Point Clouds


### Plotly visualization of Point Clouds
