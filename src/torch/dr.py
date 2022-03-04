# builtin
import os
import pathlib
import sys

# 3rd party
import numpy as np
import torch
import nvdiffrast.torch as dr
from PIL import Image
import imageio

# local
import data
import utils

# -----------------------------------------------------------------



def render(glctx, mtx, pos, pos_idx, vtx_col, col_idx, resolution: int):
    pos_clip    = utils.transform_pos(mtx, pos)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    color, _    = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
    color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
    return color

def fit_mesh(max_iter, lr_base, lr_ramp, meshdata, display_interval, imdir, calibs):
    # create tensors? or do this already in data through parameter meshdata
    tex = meshdata.tex
    vtx = meshdata.vtx
    vtx_pos_opt = torch.tensor(vtx, dtype=torch.float32, device='cuda', requires_grad=True)

    tex = torch.from_numpy(tex.astype(np.float32)).cuda()
    tex_opt = torch.full(tex.shape, 0.2, device='cuda', requires_grad=True)

    # context
    glctx = dr.RasterizeGLContext()

    optimizer = torch.optim.Adam([tex_opt], lr=lr_base)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_ramp**(float(x)/float(max_iter)))

    cams = os.listdir(imdir)
    for cam in cams:
        # get camera calibration
        calib = calibs[cam.split("_")[1]]
        intr = calib['intrinsic']
        dist = calib['distortion']
        rot = calib['rotation']
        trans = calib['translation']

        camdir = os.path.join(imdir, cam)
        frames = os.listdir(camdir)
        for frame in frames:
            # reference image to render against
            img = np.array(Image.open(os.path.join(camdir, frame)))
            colour = torch.from_numpy(img).cuda()

            # render
            for it in range(max_iter + 1):

                # projection
                r_mv = np.asarray(calib['rotation'])

                # render
                color = render(glctx, r_mvp, vtx_pos, pos_idx, vtx_col, col_idx, resolution)
                color_opt = render(glctx, r_mvp, vtx_pos_opt, pos_idx, vtx_col_opt, col_idx, resolution)

                # Compute loss and train.
                loss = torch.mean((color - color_opt) ** 2)  # L2 pixel loss.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()

                # Show/save image.
                display_image = display_interval and (it % display_interval == 0)
                if display_image:
                    img_b = color[0].cpu().numpy()
                    img_o = color_opt[0].detach().cpu().numpy()
                    img_d = render(glctx, a_mvp, vtx_pos_opt, pos_idx, vtx_col_opt, col_idx, display_res)[0]
                    img_r = render(glctx, a_mvp, vtx_pos, pos_idx, vtx_col, col_idx, display_res)[0]

                    scl = display_res // img_o.shape[0]
                    img_b = np.repeat(np.repeat(img_b, scl, axis=0), scl, axis=1)
                    img_o = np.repeat(np.repeat(img_o, scl, axis=0), scl, axis=1)
                    result_image = make_grid(np.stack([img_o, img_b, img_d.detach().cpu().numpy(), img_r.cpu().numpy()]))
                    utils.display_image(result_image, size=display_res, title='%d / %d' % (it, max_iter))