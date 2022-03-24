import os
import pathlib
import sys
import numpy as np
import torch
from PIL import Image
import json
import imageio

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
    pos_clip = camera.transformClip(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=(resolution[0], resolution[1]))

    texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
    colour = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=8)

    colour = dr.antialias(colour, rast_out, pos_clip, pos_idx)

    colour = colour * torch.clamp(rast_out[..., -1:], 0, 1)  # Mask out background.
    return colour[0]

# --------------------------------------------------------------------------------


def makeImg(arr, ncols=2):
    n, height, width, nc = arr.shape
    nrows = n//ncols
    assert n == nrows*ncols
    return arr.reshape(nrows, ncols, height, width, nc).swapaxes(1,2).reshape(height*nrows, width*ncols, nc)

# --------------------------------------------------------------------------------


def fitCube(max_iter          = 5000,
             resolution        = (1600, 1200),
             log_interval      = 10,
             display_interval  = None,
             display_res       = 512,
             out_dir           = None,
             log_fn            = None,
             basemeshpath          = "",
             imdir             = "",
             calibs            = dict(),
             texpath = None):

    mp4save_interval = 1

    # Set up logging.
    if out_dir:
        out_dir = f'{out_dir}/cube_{resolution}'
        print (f'Saving results under {out_dir}')
        writer = imageio.get_writer(f'{out_dir}/progress.mp4', mode='I', fps=30, codec='libx264', bitrate='16M')
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

    rubiks = data.MeshData(basemeshpath)
    vtxp = torch.tensor(rubiks.vertices, dtype=torch.float32, device='cuda')
    # vtxp_opt = torch.tensor(np.zeros(shape=rubiks.vertices.shape), dtype=torch.float32,
                           # device='cuda', requires_grad=True)
    vtxp_opt = torch.tensor(rubiks.vertices, dtype=torch.float32,
                            device='cuda', requires_grad=True)
    pos_idx = torch.tensor(rubiks.faces, dtype=torch.int32, device='cuda')
    uv_idx = torch.tensor(rubiks.fuv, dtype=torch.int32, device='cuda')
    uv = torch.tensor(rubiks.uv, dtype=torch.float32, device='cuda')
    texture = np.array(Image.open(texpath))
    tex = torch.tensor(texture, dtype=torch.float32, device='cuda')

    glctx = dr.RasterizeGLContext()
    optimizer = torch.optim.Adam([vtxp_opt], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.01, 10 ** (-x * 0.0005)))

    for cam in cams:
        if cam != "_pod2texture":
            continue
        # get camera calibration
        camname = cam.split("_")[1]
        calib = calibs[camname]
        intr = np.asarray(calib['intrinsic'], dtype=np.float32)
        # dist = np.asarray(calib['distortion'], dtype=np.float32)
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
                    geom_loss = torch.mean(torch.sum((torch.abs(vtxp_opt - vtxp)) ** 2, dim=0) ** 0.5)
                    avg_pos = torch.mean(vtxp_opt)
                    gl_avg.append(float(geom_loss))
                # Print/save log.
                if log_interval and (it % log_interval == 0):
                    gl_val = np.mean(np.asarray(gl_avg))
                    gl_avg = []
                    s = ""
                    s += f"iter={it}, err={gl_val}, avg_pos={avg_pos}"
                    print(s)
                    if log_file:
                        log_file.write(s + "\n")

                # split [n_vertices * 3] to [n_vertices, 3] as a view of the original tensor
                # vtxp_split = torch.reshape(vtxp, (vtxp.shape[0] // 3, 3))
                # vtxp_opt_split = torch.reshape(vtxp_opt, (vtxp_opt.shape[0]//3, 3))

                # render
                colour = render(glctx, mvp, vtxp_opt, pos_idx, uv, uv_idx, tex, resolution)

                # Compute loss and train.
                # print(f"ref: {ref.shape} --- colour: {colour.shape}")

                loss = torch.mean((ref - colour) ** 2)  # L2 pixel loss.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                display_image = display_interval and (it % display_interval == 0)
                save_mp4 = mp4save_interval and (it % mp4save_interval == 0)

                if display_image or save_mp4:
                    img_ref = ref.cpu().numpy()  # [::-1]
                    img_col = colour.detach().cpu().numpy()  # [::-1]

                    # scl = display_res // img_o.shape[0]
                    # result_image = makeImg(np.stack([img_ref, img_col]))
                    utils.display_image(img_col, size=display_res)
                    if save_mp4:
                        writer.append_data(np.clip(np.rint(img_col * 255.0), 0, 255).astype(np.uint8))

    if writer is not None:
        writer.close()
    if log_file:
        log_file.close()


#----------------------------------------------------------------------------

def main():
    # Get camera calibration
    # ...
    path = r"C:\Users\Henkka\Projects\invrend-fpc\data\calibration\2021-07-01\DI_calibration.json"
    with open(path) as json_file:
        calibs = json.load(json_file)

    # Run
    fitCube(
        max_iter=300,
        resolution=(1600, 1200),
        log_interval=100,
        display_interval=1,
        out_dir=r"C:\Users\Henkka\Projects\invrend-fpc\data\cube\out_img",
        log_fn='log.txt',
        basemeshpath=r"C:\Users\Henkka\Projects\invrend-fpc\data\cube\rubiks_bl.obj",
        imdir=r"C:\Users\Henkka\Projects\invrend-fpc\data\cube\20220310\neutrals\rubik_cube\neutral\take0001\fullres",
        calibs=calibs,
        texpath=r"C:\Users\Henkka\Projects\invrend-fpc\data\cube\text.png"
    )

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
