# 3D Graphics Systems Course - IMPA 2021

#### Leonardo Quatrin Campagnolo

---------

## Assignment 7 - Render Point Clouds and Meshes

In this assignment, the idea is to practice operations with 3D meshes and point clouds using the PyTorch3D. The code of this assignment can be found [here](https://github.com/lquatrin/i3d21/blob/main/code/a7/Assignment7.ipynb).

### Loading a textured mesh and visualizing its texture map

In this report, the cow mesh was used for the experiments, which can be loaded from PyTorch3D repository. The function **load_objs_as_meshes** can be used to load all the data from the current object. It contains a UV coordinate defined for each vertex, and we can visualize the current UV mapping using the function **texturesuv_image_matplotlib** from pytorch3d.vis.texture_vis:

<img src="data/imgs/a7/cow_mesh_texture_map.png" width="30%">

where the red dots defines the UV coordinate of each vertex.

### Rendering a Mesh

In PyTorch3D, a renderer is composed of a rasterizer and a shader which each have a number of subcomponents such as a camera (orthographic/perspective). First, the **MeshRenderer** will be used with a perspective camera and a point light, using the  **MeshRasterizer** and the **SoftPhongShader**:

<img src="data/imgs/a7/2_0.png" width="30%">

It is possible to update the rendering parameters by passing it when generating a new batch of images, such as light source position or material.

<img src="data/imgs/a7/2_1.png" width="30%"><img src="data/imgs/a7/2_3.png" width="30%">

Also changing the shader, it is possible to generate different effects. Here we have another result of the cow mesh using a new **MeshRenderer** with **HardFlatShader**:

<img src="data/imgs/a7/2_2.png" width="30%">

Some types of shader require some data to be presented in the **Meshes** object. When using **SoftGouraudShader**, the **textures** parameter must be a **TexturesVertex** type, which differs from using the UV mapping (**TexturesUV** type). In this case, we specify the current color value attached to each vertex as its estimated normal. The cow mesh already have a normal list attached to each vertex. However, we can also estimate them by averaging the normals of the faces that shares the same vertex (having a considerable smooth surface mesh). Normalizing them with **(normals+1.0) * 0.5**, i got the following result:

<img src="data/imgs/a7/gradient.png" width="30%">

### Moving the scene

As mentioned before, we can change the light source or material parameter to update the current renderer. It is also possible to change the current camera. For each new one, we use the **look_at_view_transform** method, which receives 3 parameters: the distance of the camera from the object, the elevation and the azimutal angle. Changing these parameters will generate different images of the current mesh:

<img src="data/imgs/a7/3_1.png" width="30%">

It is also possible to move the current object. Here, we create a new mesh by passing a new list of vertices rotated around the Y-axis:

```python
from pytorch3d.transforms import axis_angle_to_matrix
import math
a_angle = torch.Tensor([0, math.pi, 0])
rotate_transform = Rotate(axis_angle_to_matrix(a_angle))

verts  = mesh.verts_list()[0].clone().detach()
verts = rotate_transform.transform_points(verts)

cow_mesh = Meshes(verts=[verts],
                  faces=mesh.faces_list(),
                  textures=mesh.textures)
```

Rotating the object will consequently change the lighting effect, since the light source stayed in the same position, which can be noted by the following image:

<img src="data/imgs/a7/3_2.png" width="30%">

### Batched Rendering

Using the **look_at_view_transform** method, it is possible to create a batch of different cameras to visualize the mesh at different viewpoints:

![Batch of cows](data/imgs/a7/3_0.png)

Here we have two different views, each one rendering the current mesh in a different viewport:

<img src="data/imgs/a7/4_1.png" width="30%">

If we want to create a single object of **Meshes** with two objects, we use the method **join_meshes_as_scene**, and render the scene again:

```python
offset1 = mesh.verts_padded().new_tensor([0, 0, -1]).expand(mesh.verts_packed().shape)
offset2 = mesh.verts_padded().new_tensor([0, 0, 1]).expand(mesh.verts_packed().shape)
double_mesh = join_meshes_as_scene([cow_mesh.clone().offset_verts_(offset1), mesh.clone().offset_verts_(offset2)])

R, T = look_at_view_transform(dist=4, elev=0.0, azim=90)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

lights.location = torch.tensor([[0.0, 0.0, -3.0]], device=device)
images = renderer(double_mesh, cameras=cameras, lights=lights)
```

Using our previously rotated cow_mesh, we get the following result:

<img src="data/imgs/a7/4_2.png" width="30%">

We can note how the lighting effects are different at each mesh.

### Rendering Point Clouds

Pytorch3D also provide an option to render point clouds, by creating a **PointClouds** object. Each object receives a list of vertex positions and a list of features, such as the rgb color attached to each point. In this case, a **PointsRasterizer** is created and passed as a parameter to a **PointsRenderer**:

<img src="data/imgs/a7/5_0.png" width="30%">

There are a few parameters that can be modified, such as background color and the compositor function that blends the points rendered at each pixel. 

<img src="data/imgs/a7/5_1.png" width="30%"><img src="data/imgs/a7/5_2.png" width="30%">

#### Pulsar Backend

The pulsar backend [1] is also available in PyTorch3D. In this case, the scene is represented by a set of spheres, each being defined by a position, a radius, an opacity and a feature vector, which can correspond to lighting properties or an abstract
feature representation for use in a neural network.

<img src="data/imgs/a7/5_3.png" width="30%">

In this case, i use the orthographic projection, changing the current camera to **FoVOrthographicCameras**. Otherwise, only the first image will be rendered correctly, and the other will not render the point cloud.

<img src="data/imgs/a7/5_3_batch.png" width="30%">

### Plotly visualization of Point Clouds

It is possible to render point clouds using the avaiable methods from **pytorch3d.vis.plotly_vis**. First, it is possible to render the point cloud using the function **plot_scene**:

<img src="data/imgs/a7/6_1.png" width="100%">

It is possible to pass a batch of points for the **PointClouds** object, similar to **Meshes**. In this case, i made a simple test by rendering a batch of 2 point clouds:

<img src="data/imgs/a7/6_2.png" width="100%">

Finally, using the **plot_batch_individually** method, it is possible to render each point cloud:

<img src="data/imgs/a7/6_3.png" width="100%">

### References

[1] Lassner, C. and Zollhöfer, M., “Pulsar: Efficient Sphere-based Neural Rendering”, arXiv e-prints, 2020. URL: https://arxiv.org/pdf/2004.07484.pdf
