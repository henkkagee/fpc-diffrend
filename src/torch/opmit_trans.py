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
import os

# ---------------------------------------------------------------------------------------

def simple_render():
    """
    Render a simple scene for verifying object data.

    :return:
    """
    camdir = r"C:\Users\Henrik\fpc-diffrend\data\reference\neutral\ilkvil_neutral_edit_background\range04_pod2texture"
    frame = r"range04_pod2texture_0.tif"
    path = r"C:\Users\Henrik\fpc-diffrend\calibration\combined\calibration.json"
    with open(path) as json_file:
        calibs = json.load(json_file)
    calib = calibs['pod2texture']
    intr = np.asarray(calib['intrinsic'], dtype=np.float32)
    dist = np.asarray(calib['distortion'], dtype=np.float32)
    rot = np.asarray(calib['rotation'], dtype=np.float32)
    trans = np.asarray(calib['translation'], dtype=np.float32)
    y_opt = torch.tensor([0.0], dtype=torch.float32, device='cuda', requires_grad=True)
    fx_opt = torch.tensor([intr[0,0]], dtype=torch.float32, device='cuda', requires_grad=True)
    fy_opt = torch.tensor([intr[1,1]], dtype=torch.float32, device='cuda', requires_grad=True)

    rubiks = data.MeshData(r"C:\Users\Henrik\fpc-diffrend\data\basemesh.obj")
    # rubiks = data.MeshData(r"C:\Users\Henrik\fpc-diffrend\data\cube\rubiks.obj")

    pos = torch.tensor(rubiks.vertices, dtype=torch.float32, device='cuda')
    tri = torch.tensor(rubiks.faces, dtype=torch.int32, device='cuda')
    uv = torch.tensor(rubiks.uv, dtype=torch.float32, device='cuda')
    uv_idx = torch.tensor(rubiks.fuv, dtype=torch.int32, device='cuda')
    # texture = np.array(Image.open(r"C:\Users\Henrik\fpc-diffrend\data\cube\rubiks.png"))/255.0
    texture = np.array(Image.open(r"C:\Users\Henrik\fpc-diffrend\data\ilkvil_tex_grid_bright.png")) / 255.0
    tex = torch.tensor(texture, dtype=torch.float32, device='cuda')
    tex = tex.reshape((1600, 1600, 1))

    proj = torch.tensor(camera.intrinsic_to_projection(intr), dtype=torch.float32, device='cuda')
    # proj = torch.tensor(camera.default_projection(), dtype=torch.float32, device='cuda')
    mv = torch.tensor(camera.extrinsic_to_modelview(rot, trans), dtype=torch.float32, device='cuda')
    # mv = torch.tensor(camera.default_modelview(), dtype=torch.float32, device='cuda')

    # reference image to render against
    img = np.array(Image.open(os.path.join(camdir, frame)))/255
    ref = torch.from_numpy(img).cuda().reshape((1600, 1200, 1))
    # ref = torch.from_numpy(np.flip(img, 0).copy()).cuda().reshape((1600, 1200, 1))

    glctx = dr.RasterizeGLContext(device='cuda')
    optimizer = torch.optim.Adam([y_opt, fx_opt, fy_opt], lr=1e-1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.01, 10 ** (-x * 0.0005)))
    bestloss = 1000000
    bestvals = {}

    for i in range(5000):
        transvec = torch.zeros(3, dtype=torch.float32, device='cuda')
        transvec[1] = y_opt
        tr = torch.matmul(camera.translate_tensor(transvec),
                          torch.tensor(camera.rotate_x(0.0), dtype=torch.float32, device='cuda'))
        # tr = torch.matmul(tr, torch.tensor(camera.rotate_y(-0.6), dtype=torch.float32, device='cuda'))
        tp = torch.matmul(mv, tr)
        pos_split = torch.reshape(pos, (pos.shape[0] // 3, 3))
        proj = torch.tensor(camera.intrinsic_to_projection(intr), dtype=torch.float32, device='cuda')
        proj[(0), (0)] = fx_opt / intr[0,2]
        proj[(1), (1)] = fy_opt / intr[1,2]
        mvp = torch.matmul(proj, tp)

        pos_clip = camera.transform_clip(mvp, pos_split)
        # print(f"i[{i}]\npos_clip: {pos_clip}")

        rast, _ = dr.rasterize(glctx, pos_clip, tri, resolution=[1600, 1200])
        texc, _ = dr.interpolate(uv[None, ...], rast, uv_idx)
        colour = dr.texture(tex[None, ...], texc, filter_mode='linear')
        colour = dr.antialias(colour, rast, pos_clip, tri)
        colour = torch.where(rast[..., 3:] > 0, colour, torch.tensor(0.0).cuda())[0]

        # Compute loss and train.
        # geom_loss = torch.mean(torch.sum((torch.abs(vtx_pos_opt) - .5) ** 2, dim=1) ** 0.5)
        # print(f"ref: {ref.shape}\ncolour: {colour.shape}")
        loss = torch.mean((ref - colour) ** 2)  # L2 pixel loss.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss < bestloss:
            bestvals = {"trans_opt": y_opt.cpu().detach().numpy()[0],
                        "fx": fx_opt.cpu().detach().numpy()[0], "fy": fy_opt.cpu().detach().numpy()[0]}
            bestloss = loss

        # undistort w.r.t. lens distortion
        # mapx, mapy = cv2.initUndistortRectifyMap(intr, dist, np.empty(0), intr, (1600, 1200), cv2.CV_32F)

        # img = cv2.undistort(img, intr, dist)
        img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
        # utils.save_image("test.png", img)
        if i % 5 == 0:
            print(f"i: {i}, ref0: {ref[600][800]}, col0: {colour[600][800]}\n bestvals: {bestvals}, loss: {loss}")
            img = colour.cpu().detach().numpy()[::-1, :, :]  # Flip vertically due to opengl standards
            utils.display_image(img)

    print(bestvals)
    # utils.save_image("test.png", img)


        # imageio.imsave('face.png', img)

# ---------------------------------------------------------------------------------------


def main():
    simple_render()

# ---------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()