import os
import torch
import matplotlib.pyplot as plt
from skimage.io import imread

from pytorch3d.utils import ico_sphere
import numpy as np
from tqdm.notebook import tqdm

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj

from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    look_at_rotation, 
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    HardFlatShader,
    TexturesVertex
)

def OptimizeMeshVertices(src_mesh, target_mesh, device, 
                         num_views_per_iteration=2,
                         visualize_prediction_cb=None,
                         NumberOfIterations=2000,
                         elev0 = 0,    elev1 = 360,
                         azim0 = -180, azim1 = 180,
                         num_views = 20):
  # the number of different viewpoints from which we want to render the mesh.
  #num_views = 20

  # Get a batch of viewing angles. 
  elev = torch.linspace(elev0, elev1, num_views)
  azim = torch.linspace(azim0, azim1, num_views)

  # Place a point light in front of the object. As mentioned above, the front of 
  # the cow is facing the -z direction. 
  lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

  # Initialize an OpenGL perspective camera that represents a batch of different 
  # viewing angles. All the cameras helper methods support mixed type inputs and 
  # broadcasting. So we can view the camera from the a distance of dist=2.7, and 
  # then specify elevation and azimuth angles for each viewpoint as tensors. 
  R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
  cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

  # We arbitrarily choose one particular view that will be used to visualize 
  # results
  camera = FoVPerspectiveCameras(device=device, R=R[None, 1, ...], 
                                                T=T[None, 1, ...]) 

  # Create a batch of meshes by repeating the cow mesh and associated textures. 
  # Meshes has a useful `extend` method which allows us do this very easily. 
  # This also extends the textures. 
  meshes = target_mesh.extend(num_views)

  sigma = 1e-4
  raster_settings_silhouette = RasterizationSettings(
      perspective_correct=False,
      image_size=128, 
      blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
      faces_per_pixel=50, 
  )

  # Silhouette renderer 
  renderer_silhouette = MeshRenderer(
      rasterizer=MeshRasterizer(
          cameras=camera, 
          raster_settings=raster_settings_silhouette
      ),
      shader=SoftSilhouetteShader()
  )

  # Render silhouette images.  The 3rd channel of the rendering output is 
  # the alpha/silhouette channel
  silhouette_images = renderer_silhouette(meshes, cameras=cameras, lights=lights)
  target_silhouette = [silhouette_images[i, ..., 3] for i in range(num_views)]

  # Render the cow mesh from each viewing angle
  #target_images = renderer(meshes, cameras=cameras, lights=lights)

  # Our multi-view cow dataset will be represented by these 2 lists of tensors,
  # each of length num_views.
  #target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
  target_cameras = [FoVPerspectiveCameras(device=device, R=R[None, i, ...], 
                                            T=T[None, i, ...]) for i in range(num_views)]

  # Optimize using rendered silhouette image loss, mesh edge loss, mesh normal 
  # consistency, and mesh laplacian smoothing
  losses = { 
    "silhouette": {"weight": 1.0,  "values": []},
    "edge":       {"weight": 1.0,  "values": []},
    "normal":     {"weight": 0.01, "values": []},
    "laplacian":  {"weight": 1.0,  "values": []},
  }

  # Number of optimization steps
  Niter = NumberOfIterations
  # Plot period for the losses
  plot_period = 250

  verts_shape = src_mesh.verts_packed().shape
  deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)

  # The optimizer
  optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

  loop = tqdm(range(Niter))

  for i in loop:
      # Initialize optimizer
      optimizer.zero_grad()
      
      # Deform the mesh
      new_src_mesh = src_mesh.offset_verts(deform_verts)
      
      # Losses to smooth /regularize the mesh shape
      loss = {k: torch.tensor(0.0, device=device) for k in losses}

      #update_mesh_shape_prior_losses(new_src_mesh, loss)
      #####################################################
      # and (b) the edge length of the predicted mesh
      loss["edge"] = mesh_edge_loss(new_src_mesh)
      # mesh normal consistency
      loss["normal"] = mesh_normal_consistency(new_src_mesh)
      # mesh laplacian smoothing
      loss["laplacian"] = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
      #####################################################

      # Compute the average silhouette loss over two random views, as the average 
      # squared L2 distance between the predicted silhouette and the target 
      # silhouette from our dataset
      for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
          images_predicted = renderer_silhouette(new_src_mesh, cameras=target_cameras[j], lights=lights)
          predicted_silhouette = images_predicted[..., 3]
          loss_silhouette = ((predicted_silhouette - target_silhouette[j]) ** 2).mean()
          loss["silhouette"] += loss_silhouette / num_views_per_iteration
      
      # Weighted sum of the losses
      sum_loss = torch.tensor(0.0, device=device)
      for k, l in loss.items():
          sum_loss += l * losses[k]["weight"]
          losses[k]["values"].append(l)
      
      # Print the losses
      loop.set_description("total_loss = %.6f" % sum_loss)
      
      # Plot mesh
      if i % plot_period == 0 and visualize_prediction_cb is not None:
          visualize_prediction_cb(new_src_mesh, title="iter: %d" % i, silhouette=True,
                              target_image=target_silhouette[1])
          
      # Optimization step
      sum_loss.backward()
      optimizer.step()
  return { "losses" : losses, "new_mesh" : new_src_mesh } 

def OptimizeMeshVerticesAndTextures(src_mesh, target_mesh, device, 
                                    num_views_per_iteration=2,
                                    visualize_prediction_cb=None,
                                    NumberOfIterations=2000,
                                    LossRGB        = 1.0,
                                    LossSILHOUETTE = 1.0,
                                    LossEDGE       = 1.0,
                                    LossNORMAL     = 0.01,
                                    LossLAPLACIAN  = 1.0):
  # the number of different viewpoints from which we want to render the mesh.
  num_views = 20

  # Get a batch of viewing angles. 
  elev = torch.linspace(0, 360, num_views)
  azim = torch.linspace(-180, 180, num_views)

  # Place a point light in front of the object. As mentioned above, the front of 
  # the cow is facing the -z direction. 
  lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

  # Initialize an OpenGL perspective camera that represents a batch of different 
  # viewing angles. All the cameras helper methods support mixed type inputs and 
  # broadcasting. So we can view the camera from the a distance of dist=2.7, and 
  # then specify elevation and azimuth angles for each viewpoint as tensors. 
  R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
  cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

  # We arbitrarily choose one particular view that will be used to visualize 
  # results
  camera = FoVPerspectiveCameras(device=device, R=R[None, 1, ...], 
                                                T=T[None, 1, ...]) 

  # Create a batch of meshes by repeating the cow mesh and associated textures. 
  # Meshes has a useful `extend` method which allows us do this very easily. 
  # This also extends the textures. 
  meshes = target_mesh.extend(num_views)

  raster_settings = RasterizationSettings(
      perspective_correct=False,
      image_size=128, 
      blur_radius=0.0, 
      faces_per_pixel=1, 
  )

  # Create a phong renderer by composing a rasterizer and a shader. The textured 
  # phong shader will interpolate the texture uv coordinates for each vertex, 
  # sample from a texture image and apply the Phong lighting model
  renderer = MeshRenderer(
      rasterizer=MeshRasterizer(
          cameras=camera, 
          raster_settings=raster_settings
      ),
      shader=SoftPhongShader(
          device=device, 
          cameras=camera,
          lights=lights
      )
  )

  # Render the cow mesh from each viewing angle
  target_images = renderer(meshes, cameras=cameras, lights=lights)

  # Our multi-view cow dataset will be represented by these 2 lists of tensors,
  # each of length num_views.
  target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
  target_cameras = [FoVPerspectiveCameras(device=device, R=R[None, i, ...], 
                                                         T=T[None, i, ...]) for i in range(num_views)]

  sigma = 1e-4
  raster_settings_silhouette = RasterizationSettings(
      perspective_correct=False,
      image_size=128, 
      blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
      faces_per_pixel=50, 
  )

  # Silhouette renderer 
  renderer_silhouette = MeshRenderer(
      rasterizer=MeshRasterizer(
          cameras=camera, 
          raster_settings=raster_settings_silhouette
      ),
      shader=SoftSilhouetteShader()
  )

  # Render silhouette images.  The 3rd channel of the rendering output is 
  # the alpha/silhouette channel
  silhouette_images = renderer_silhouette(meshes, cameras=cameras, lights=lights)
  target_silhouette = [silhouette_images[i, ..., 3] for i in range(num_views)]

  ### CREATE RENDERING
  # Rasterization settings for differentiable rendering, where the blur_radius
  # initialization is based on Liu et al, 'Soft Rasterizer: A Differentiable 
  # Renderer for Image-based 3D Reasoning', ICCV 2019
  raster_settings_soft = RasterizationSettings(
      perspective_correct=False,
      image_size=128, 
      blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
      faces_per_pixel=50, 
  )

  # Differentiable soft renderer using per vertex RGB colors for texture
  renderer_textured = MeshRenderer(
      rasterizer=MeshRasterizer(
          cameras=camera, 
          raster_settings=raster_settings_soft
      ),
      shader=SoftPhongShader(device=device, 
          cameras=camera,
          lights=lights)
  )

  # Number of optimization steps
  Niter = NumberOfIterations
  # Plot period for the losses
  plot_period = 250

  ### 
  # Optimize using rendered RGB image loss, rendered silhouette image loss, mesh 
  # edge loss, mesh normal consistency, and mesh laplacian smoothing
  losses = {"rgb":        { "weight": LossRGB,        "values": []},
            "silhouette": { "weight": LossSILHOUETTE, "values": []},
            "edge":       { "weight": LossEDGE,       "values": []},
            "normal":     { "weight": LossNORMAL,     "values": []},
            "laplacian":  { "weight": LossLAPLACIAN,  "values": []},
          }
  # We will learn to deform the source mesh by offsetting its vertices
  # The shape of the deform parameters is equal to the total number of vertices in 
  # src_mesh
  verts_shape = src_mesh.verts_packed().shape
  deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)

  # We will also learn per vertex colors for our sphere mesh that define texture 
  # of the mesh
  sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=True)

  # The optimizer
  if (LossSILHOUETTE + LossEDGE + LossNORMAL + LossLAPLACIAN) > 0:
    optimizer = torch.optim.SGD([deform_verts, sphere_verts_rgb], lr=1.0, momentum=0.9)
  else:
    optimizer = torch.optim.SGD([sphere_verts_rgb], lr=1.0, momentum=0.9)

  loop = tqdm(range(Niter))

  for i in loop:
    # Initialize optimizer
    optimizer.zero_grad()
    
    # Deform the mesh
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    
    # Add per vertex colors to texture the mesh
    new_src_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb) 
    
    # Losses to smooth /regularize the mesh shape
    loss = {k: torch.tensor(0.0, device=device) for k in losses}

    #update_mesh_shape_prior_losses(new_src_mesh, loss)
    #####################################################
    # and (b) the edge length of the predicted mesh
    loss["edge"] = mesh_edge_loss(new_src_mesh)
    # mesh normal consistency
    loss["normal"] = mesh_normal_consistency(new_src_mesh)
    # mesh laplacian smoothing
    loss["laplacian"] = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
    #####################################################
    
    # Randomly select two views to optimize over in this iteration.  Compared
    # to using just one view, this helps resolve ambiguities between updating
    # mesh shape vs. updating mesh texture
    for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
      images_predicted = renderer_textured(new_src_mesh, cameras=target_cameras[j], lights=lights)

      # Squared L2 distance between the predicted silhouette and the target 
      # silhouette from our dataset
      predicted_silhouette = images_predicted[..., 3]
      loss_silhouette = ((predicted_silhouette - target_silhouette[j]) ** 2).mean()
      loss["silhouette"] += loss_silhouette / num_views_per_iteration
        
      # Squared L2 distance between the predicted RGB image and the target 
      # image from our dataset
      predicted_rgb = images_predicted[..., :3]
      loss_rgb = ((predicted_rgb - target_rgb[j]) ** 2).mean()
      loss["rgb"] += loss_rgb / num_views_per_iteration
    
    # Weighted sum of the losses
    sum_loss = torch.tensor(0.0, device=device)
    for k, l in loss.items():
      sum_loss += l * losses[k]["weight"]
      losses[k]["values"].append(l)
    
    # Print the losses
    loop.set_description("total_loss = %.6f" % sum_loss)
    
    # Plot mesh
    if i % plot_period == 0 and visualize_prediction_cb is not None:
      visualize_prediction_cb(new_src_mesh, renderer=renderer_textured, title="iter: %d" % i, silhouette=False)
        
    # Optimization step
    sum_loss.backward()
    optimizer.step()
  return { "losses" : losses, "new_mesh" : new_src_mesh } 




