import imageio
import numpy as np
import torch
import nvdiffrast.torch as dr
import data
from PIL import Image
import camera
import utils
from time import sleep
import pathlib

# ---------------------------------------------------------------------------------------

def simpleRender():
    rubiks = data.MeshData(r"C:\Users\Henkka\Projects\invrend-fpc\data\cube\simplecube.obj")

    pos = torch.tensor(rubiks.vertices, dtype=torch.float32, device='cuda')
    tri = torch.tensor(rubiks.faces, dtype=torch.int32, device='cuda')
    uv = torch.tensor(rubiks.uv, dtype=torch.float32, device='cuda')
    uv_idx = torch.tensor(rubiks.fuv, dtype=torch.int32, device='cuda')
    texture = np.array(Image.open(r"C:\Users\Henkka\Projects\invrend-fpc\data\cube\textest.png"))/255.0
    tex = torch.tensor(texture, dtype=torch.float32, device='cuda')

    proj = camera.intrinsicToProjection()
    mv = camera.extrinsicToModelview()

    glctx = dr.RasterizeGLContext(device='cuda')
    a = 0
    for i in range(100):
        rot = np.matmul(camera.rotate_x(a), camera.rotate_y(a))
        mvr = np.matmul(mv, rot)
        a += 0.01 * 2 * np.pi
        mvp = np.matmul(proj, mvr).astype(np.float32)

        pos_clip = camera.transformClip(mvp, pos)

        rast, _ = dr.rasterize(glctx, pos_clip, tri, resolution=[1024, 1024])
        texc, _ = dr.interpolate(uv[None, ...], rast, uv_idx)
        colour = dr.texture(tex[None, ...], texc, filter_mode='linear')
        colour = dr.antialias(colour, rast, pos_clip, tri)

        img = colour.cpu().numpy()[0, ::-1, :, :] # Flip vertically.
        img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8

        # print("Saving to 'test.png'.")
        # imageio.imsave('test.png', img)
        utils.display_image(img)
        # sleep(0.3)

# ---------------------------------------------------------------------------------------

def npz():
    datadir = f'{pathlib.Path(__file__).absolute().parents[2]}/data'
    print(f"datadir: {datadir}")
    with np.load(f'{datadir}/test/earth.npz') as f:
        pos_idx, pos, uv_idx, uv, tex = f.values()
    tex = tex.astype(np.float32)/255.0

    print(f"shapes:\npos: {pos.shape}\nuv: {uv.shape}\npos_idx: {pos_idx.shape}\nuv_idx: {uv_idx.shape}\n")

    with open("earth.obj", 'w') as f:
        for v in pos:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for t in uv:
            f.write(f"vt {t[0]} {t[1]}\n")
        for i in range(len(pos_idx)):
            f.write(f"f {pos_idx[i][0]+1}/{uv_idx[i][0]+1} {pos_idx[i][1]+1}/{uv_idx[i][1]+1} {pos_idx[i][2]+1}/{uv_idx[i][2]+1}\n")

# ---------------------------------------------------------------------------------------

def main():
    # npz()
    simpleRender()

# ---------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()