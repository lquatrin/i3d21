import torch
import pytorch3d
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene, Pointclouds
from pytorch3d.io import load_obj, save_obj

def render2meshplotly(source_mesh, target_mesh):
  fig = plot_scene({
      "Source": {
          "mesh1": source_mesh
      },
      "Target": {
          "mesh1": target_mesh
      }
    },
    ncols=2,
    zaxis={"backgroundcolor":"rgb(200, 230, 200)"},
    xaxis={"backgroundcolor":"rgb(200, 200, 230)"},
    yaxis={"backgroundcolor":"rgb(230, 200, 200)"},
    axis_args=AxisArgs(showgrid=True)
  )
  fig.show()

def rendermeshplotly(smesh):
  fig = plot_scene({
      "Mesh": {
          "mesh1": smesh
      },
    },
    ncols=2,
    zaxis={"backgroundcolor":"rgb(200, 230, 200)"},
    xaxis={"backgroundcolor":"rgb(200, 200, 230)"},
    yaxis={"backgroundcolor":"rgb(230, 200, 200)"},
    axis_args=AxisArgs(showgrid=True)
  )
  fig.show()


def normalizeverts(verts):
  # dim (int or tuple of python:ints) – the dimension or dimensions to reduce.
  center = verts.mean(dim=0)#

  # translate to the center of the vertices
  verts = verts - center

  # get maximum value at each dimension
  #scale = max(verts.abs().max(0)[0])
  scale = verts.abs().max()
  normalized_verts = verts / scale
  return normalized_verts, center, scale

def loadandnormalizeobj(obj_path, device):
  # We read the target 3D model using load_obj
  verts, faces, aux = load_obj(obj_path)

  # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
  # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
  # For this tutorial, normals and textures are ignored.
  faces_idx = faces.verts_idx.to(device)
  verts = verts.to(device)

  normalized_verts, center, scale = normalizeverts(verts)
  obj_mesh = Meshes(verts=[normalized_verts], faces=[faces_idx])
  return obj_mesh, center, scale