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
import cv2

# ---------------------------------------------------------------------------------------

def simple_render():
    """
    Render a simple scene for verifying object data.

    :return:
    """
    path = r"C:\Users\Henkka\Projects\invrend-fpc\data\calibration\combined\calibration.json"
    with open(path) as json_file:
        calibs = json.load(json_file)
    calib = calibs['pod3secondary']
    intr = np.asarray(calib['intrinsic'], dtype=np.float32)
    dist = np.asarray(calib['distortion'], dtype=np.float32)
    rot = np.asarray(calib['rotation'], dtype=np.float32)
    trans = np.asarray(calib['translation'], dtype=np.float32)

    rubiks = data.MeshData(r"C:\Users\Henkka\Projects\invrend-fpc\data\basemesh.obj")
    # rubiks = data.MeshData(r"C:\Users\Henkka\Projects\invrend-fpc\data\cube\rubiks.obj")

    pos = torch.tensor(rubiks.vertices, dtype=torch.float32, device='cuda')
    tri = torch.tensor(rubiks.faces, dtype=torch.int32, device='cuda')
    uv = torch.tensor(rubiks.uv, dtype=torch.float32, device='cuda')
    uv_idx = torch.tensor(rubiks.fuv, dtype=torch.int32, device='cuda')
    # texture = np.array(Image.open(r"C:\Users\Henkka\Projects\invrend-fpc\data\cube\rubiks.png"))/255.0
    texture = np.array(Image.open(r"C:\Users\Henkka\Projects\invrend-fpc\data\ilkvil_tex_grid.png")) / 255.0
    tex = torch.tensor(texture, dtype=torch.float32, device='cuda')

    proj = torch.tensor(camera.intrinsic_to_projection(intr), dtype=torch.float32, device='cuda')
    # proj = torch.tensor(camera.default_projection(), dtype=torch.float32, device='cuda')
    mv = torch.tensor(camera.extrinsic_to_modelview(rot, trans), dtype=torch.float32, device='cuda')
    # mv = torch.tensor(camera.default_modelview(), dtype=torch.float32, device='cuda')

    glctx = dr.RasterizeGLContext(device='cuda')
    for i in range(200):

        tr = torch.matmul(torch.tensor(camera.translate(0.0, 0.0, 0.0), dtype=torch.float32, device='cuda'),
                          torch.tensor(camera.rotate_x(0.0), dtype=torch.float32, device='cuda'))
        # tr = torch.matmul(tr, torch.tensor(camera.rotate_y(-0.6), dtype=torch.float32, device='cuda'))
        tp = torch.matmul(mv, tr)
        pos_split = torch.reshape(pos, (pos.shape[0] // 3, 3))
        proj[(0), (0)] = torch.tensor(intr[(0), (0)], dtype=torch.float32, device='cuda') / intr[0, 2]
        proj[(1), (1)] = torch.tensor((intr[(1), (1)]), dtype=torch.float32, device='cuda') / intr[1, 2]
        mvp = torch.matmul(proj, tp)

        pos_clip = camera.transform_clip(mvp, pos_split)
        pos_clip = pos_clip
        print(f"i[{i}]\npos_clip: {pos_clip}")

        rast, _ = dr.rasterize(glctx, pos_clip, tri, resolution=[1600, 1200])
        texc, _ = dr.interpolate(uv[None, ...], rast, uv_idx)
        colour = dr.texture(tex[None, ...], texc, filter_mode='linear')
        colour = dr.antialias(colour, rast, pos_clip, tri)
        colour = torch.where(rast[..., 3:] > 0, colour, torch.tensor(0.0).cuda())

        img = colour.cpu().numpy()[0, ::-1, :, :] # Flip vertically due to opengl standards

        # undistort w.r.t. lens distortion
        # mapx, mapy = cv2.initUndistortRectifyMap(intr, dist, np.empty(0), intr, (1600, 1200), cv2.CV_32F)


        # img = cv2.undistort(img, intr, dist)
        img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
        utils.save_image("test.png", img)
        utils.display_image(img)
        return

        # imageio.imsave('face.png', img)

# ---------------------------------------------------------------------------------------


def main():
    simple_render()

# ---------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()