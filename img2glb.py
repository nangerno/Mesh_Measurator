import os
import numpy as np
import trimesh
from PIL import Image
from zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
from zoedepth.utils.geometry import depth_to_points, create_triangles
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

def depth_edges_mask(depth):
    """Returns a mask of edges in the depth map."""
    depth_dx, depth_dy = np.gradient(depth)
    depth_grad = np.sqrt(depth_dx ** 2 + depth_dy ** 2)
    return depth_grad > 0.05

def get_mesh(model, image_path, output_path, keep_edges=False):
    """Generate a 3D mesh from an input image and save as GLB."""
    # Load and resize image
    image = Image.open(image_path)
    image.thumbnail((1024, 1024))  # limit the size of the input image
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    depth = model.infer_pil(image)
    pts3d = depth_to_points(depth[None])
    pts3d = pts3d.reshape(-1, 3)
    verts = pts3d.reshape(-1, 3)
    image_array = np.array(image)
    h, w = depth.shape
    if keep_edges:
        triangles = create_triangles(h, w)
    else:
        triangles = create_triangles(h, w, mask=~depth_edges_mask(depth))
    colors = image_array.reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=verts, faces=triangles, vertex_colors=colors)
    mesh.export(output_path)
    print(f"Mesh saved to {output_path}")

def load_zoe_model():
    conf = get_config("zoedepth_nk", "infer")
    print("this model's config ========>", conf)
    model = build_model(conf)
    return model