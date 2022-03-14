import os
import pathlib
import sys
import numpy as np
import torch
from PIL import Image

import src.torch.data as data
import src.torch.utils as utils
import src.torch.camera as camera

import nvdiffrast.torch as dr

# --------------------------------------------------------------------------------


def assertNumFrames(cams, imdir):
    """
    Get number of frames outside optimization loop to init tensors and make sure each camera has the
    same number of frames

    :param cams: List of camera directories with stored reference frames
    :return:
    """
    n_frames = []
    for cam in cams:
        camdir = os.path.join(imdir, cam)
        frames = os.listdir(camdir)
        n_frames.append(len(frames))
    assert not any([x != n_frames[0] for x in n_frames])
    return n_frames[0]

# --------------------------------------------------------------------------------


def render(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution: tuple):
    pos_clip    = camera.transformClip(mtx, pos)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution[0], resolution[1]])
    texc, _    = dr.interpolate(uv[None, ...], rast_out, uv_idx)
    colour = dr.texture(tex[None, ...], texc, filter_mode='linear')
    color       = dr.antialias(colour, rast_out, pos_clip, pos_idx)
    return color

# --------------------------------------------------------------------------------


def makeImg(arr, ncols=2):
    n, height, width, nc = arr.shape
    nrows = n//ncols
    assert n == nrows*ncols
    return arr.reshape(nrows, ncols, height, width, nc).swapaxes(1,2).reshape(height*nrows, width*ncols, nc)

# --------------------------------------------------------------------------------


def fitCube(max_iter          = 5000,
             resolution        = (1200, 1600),
             log_interval      = 10,
             display_interval  = None,
             display_res       = 512,
             out_dir           = None,
             log_fn            = None,
             imdir             = "",
             calibs            = dict()):

    # Set up logging.
    if out_dir:
        out_dir = f'{out_dir}/cube_{resolution}'
        print (f'Saving results under {out_dir}')
    else:
        out_dir = None
        print ('No output directory specified, not saving log or images')

    cams = os.listdir(imdir)
    n_frames = assertNumFrames(cams, imdir)

    gl_avg = []
    log_file = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if log_fn:
            log_file = open(f'{out_dir}/{log_fn}', 'wt')

    rubikspath = ""
    rubiks = data.MeshData(rubikspath)
    vtxp = torch.tensor(rubiks.vertices, type=torch.float32, device='cuda')
    vtxp_opt = torch.tensor(np.zeros(shape=rubiks.vertices.shape), type=torch.float32,
                           device='cuda', requires_grad=True)
    pos_idx = torch.tensor(rubiks.faces, type=torch.float32, device='cuda')
    uv_idx = torch.tensor(rubiks.fuv, type=torch.float32, device='cuda')
    uv = torch.tensor(rubiks.uv, type=torch.float32, device='cuda')
    tex = torch.full(uv.shape, 0.2, device='cuda')

    glctx = dr.RasterizeGLContext()
    optimizer = torch.optim.Adam([vtxp_opt], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.01, 10 ** (-x * 0.0005)))

    for cam in cams:
        # get camera calibration
        camname = cam.split("_")[1]
        calib = calibs[camname]
        intr = np.asarray(calib['intrinsic'], dtype=np.float32)
        dist = np.asarray(calib['distortion'], dtype=np.float32)
        rot = np.asarray(calib['rotation'], dtype=np.float32)
        trans = np.asarray(calib['translation'], dtype=np.float32)

        camdir = os.path.join(imdir, cam)
        frames = os.listdir(camdir)
        for frame in frames:
            # reference image to render against
            img = np.array(Image.open(os.path.join(camdir, frame)))
            ref = torch.from_numpy(img).cuda()
            # lens distortion currently handled as preprocess in reference images
            projection = camera.intrinsicToProjection(intr)
            modelview = camera.extrinsicToModelview(rot, trans)
            mvp = np.matmul(projection, modelview).astype(np.float32)

            for it in range(max_iter + 1):
                # rotation/translation matrix for offsetting object so that it matches the image
                # camera distortion is handled as preprocessing step on the reference images

                # geometric error
                with torch.no_grad():
                    geom_loss = torch.mean(torch.sum((torch.abs(vtxp_opt) - vtxp) ** 2, dim=1) ** 0.5)
                    gl_avg.append(float(geom_loss))
                # Print/save log.
                if log_interval and (it % log_interval == 0):
                    gl_val = np.mean(np.asarray(gl_avg))
                    gl_avg = []
                    s = ""
                    s += "iter=%d,err=%f" % (it, gl_val)
                    print(s)
                    if log_file:
                        log_file.write(s + "\n")

                colour = render(glctx, mvp, vtxp_opt, pos_idx, uv, uv_idx, tex, resolution)

                # Compute loss and train.
                loss = torch.mean((ref - colour) ** 2)  # L2 pixel loss.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                display_image = display_interval and (it % display_interval == 0)
                if display_image:
                    img_ref = ref[0].cpu().numpy()[::-1]
                    img_col = colour[0].detach().cpu().numpy()[::-1]

                    # scl = display_res // img_o.shape[0]
                    result_image = makeImg(np.stack([img_ref, img_col]))
                    utils.display_image(result_image, size=display_res)

    if log_file:
        log_file.close()


#----------------------------------------------------------------------------

def main():
    # Get camera calibration
    # ...

    # Run
    fitCube(
        max_iter=1000,
        resolution=(1200, 1600),
        log_interval=10,
        display_interval=50,
        out_dir=r"W:\git\fpc-diffrend\data\inferred\cube\out_img",
        log_fn='log.txt',
        imdir=r"\\rmd.remedy.fi\Capture\Northlight\RAW\20220310\neutrals\rubik_cube\neutral\take001"
    )

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
