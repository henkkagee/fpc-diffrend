import imageio
import numpy as np
import torch
import nvdiffrast.torch as dr
import src.torch.data as data
from PIL import Image
import camera
import utils
from time import sleep
import pathlib
import json

# ---------------------------------------------------------------------------------------

def simple_render():
    """
    Render a simple scene for verifying object data.

    :return:
    """
    path = r"C:\Users\Henkka\Projects\invrend-fpc\data\calibration\2021-07-01\DI_calibration.json"
    with open(path) as json_file:
        calibs = json.load(json_file)
    calib = calibs['pod2texture']
    intr = np.asarray(calib['intrinsic'], dtype=np.float32)
    # dist = np.asarray(calib['distortion'], dtype=np.float32)
    rot = np.asarray(calib['rotation'], dtype=np.float32)
    trans = np.asarray(calib['translation'], dtype=np.float32)

    rubiks = data.MeshData(r"C:\Users\Henkka\Projects\invrend-fpc\data\cube\rubiks.obj")

    pos = torch.tensor(rubiks.vertices, dtype=torch.float32, device='cuda')
    tri = torch.tensor(rubiks.faces, dtype=torch.int32, device='cuda')
    uv = torch.tensor(rubiks.uv, dtype=torch.float32, device='cuda')
    uv_idx = torch.tensor(rubiks.fuv, dtype=torch.int32, device='cuda')
    texture = np.array(Image.open(r"C:\Users\Henkka\Projects\invrend-fpc\data\cube\tex.png"))/255.0
    tex = torch.tensor(texture, dtype=torch.float32, device='cuda')

    """proj = camera.default_projection()
    mv = camera.default_modelview()"""
    proj = camera.intrinsic_to_projection(intr)
    mv = camera.extrinsic_to_modelview(rot, trans)

    glctx = dr.RasterizeGLContext(device='cuda')
    a = 0
    for i in range(100):
        rot = np.matmul(camera.rotate_x(0.2), camera.rotate_y(a))
        mvr = np.matmul(mv, rot)
        a += 0.01 * 2 * np.pi
        mvp = np.matmul(proj, mvr).astype(np.float32)

        pos_clip = camera.transform_clip(mvp, pos)

        rast, _ = dr.rasterize(glctx, pos_clip, tri, resolution=[1600, 1200])
        texc, _ = dr.interpolate(uv[None, ...], rast, uv_idx)
        colour = dr.texture(tex[None, ...], texc, filter_mode='linear')
        colour = dr.antialias(colour, rast, pos_clip, tri)

        img = colour.cpu().numpy()[0, ::-1, :, :] # Flip vertically.
        img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8

        utils.display_image(img)
        # imageio.imsave('cube.png', img)

# ---------------------------------------------------------------------------------------


def main():
    simple_render()

# ---------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()