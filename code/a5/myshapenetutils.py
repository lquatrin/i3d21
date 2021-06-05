import torch
import pytorch3d
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene, Pointclouds
from pytorch3d.io import load_obj, save_obj

def render2pointcloudplotly(pt1, pt2, title1 = "Source", title2 = "Target"):
  fig = plot_scene({
      title1: {
          "mesh1": pt1
      },
      title2: {
          "mesh2": pt2
      }
    },
    ncols=2,
    xaxis={"backgroundcolor":"rgb(200, 200, 230)"},
    yaxis={"backgroundcolor":"rgb(230, 200, 200)"},
    zaxis={"backgroundcolor":"rgb(200, 230, 200)"},
    axis_args=AxisArgs(showgrid=True)
  )
  fig.show()

def render2meshplotly(mesh1, mesh2, title1 = "Source", title2 = "Target"):
  fig = plot_scene({
      title1: {
          "mesh1": mesh1
      },
      title2: {
          "mesh2": mesh2
      }
    },
    ncols=2,
    xaxis={"backgroundcolor":"rgb(200, 200, 230)"},
    yaxis={"backgroundcolor":"rgb(230, 200, 200)"},
    zaxis={"backgroundcolor":"rgb(200, 230, 200)"},
    axis_args=AxisArgs(showgrid=True)
  )
  fig.show()

def rendermeshplotly(mesh, title = "Mesh"):
  fig = plot_scene({
      title: {
          "mesh1": mesh
      },
    },
    xaxis={"backgroundcolor":"rgb(200, 200, 230)"},
    yaxis={"backgroundcolor":"rgb(230, 200, 200)"},
    zaxis={"backgroundcolor":"rgb(200, 230, 200)"},
    axis_args=AxisArgs(showgrid=True)
  )
  fig.show()
