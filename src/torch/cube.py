import os
import pathlib
import sys
import json

import numpy as np
import torch
import roma
from PIL import Image
import imageio
import nvdiffrast.torch as dr

import src.torch.data as data
import src.torch.utils as utils
import src.torch.camera as camera

# -------------------------------------------------------------------------------------------------


def assert_num_frames(cams, imdir):
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

# -------------------------------------------------------------------------------------------------


def render(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution: tuple):
    """
    Render an object with nvdiffrast.

    :param glctx:
    :param mtx:
    :param pos:
    :param pos_idx:
    :param uv:
    :param uv_idx:
    :param tex:
    :param resolution:
    :return:
    """
    pos_clip = camera.transform_clip(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=(resolution[0], resolution[1]))

    # texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
    texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
    # colour = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=8)
    colour = dr.texture(tex[None, ...], texc, filter_mode='linear')
    colour = dr.antialias(colour, rast_out, pos_clip, pos_idx)
    # colour = colour * torch.clamp(rast_out[..., -1:], 35.0/255.0, 36.0/255.0)
    colour = torch.where(rast_out[..., 3:] > 0, colour, torch.tensor(0.0).cuda(device='cuda:1'))
    return colour[0]

# -------------------------------------------------------------------------------------------------


def make_img(arr, ncols=2):
    """
    Stack a number of images into a grid.

    :param arr: Array of images of same shape
    :param ncols: Number of columns in image grid.
    :return:
    """
    n, height, width, nc = arr.shape
    nrows = n//ncols
    assert n == nrows*ncols
    return arr.reshape(nrows, ncols, height, width, nc).swapaxes(1,2).reshape(height*nrows, width*ncols, nc)

# -------------------------------------------------------------------------------------------------


def rot_trans_matrices(rot_tensor, t_tensor):
    """
    Get rotation and translation matrices of shape (4,4) without using torch.tensor() since that does not preserve
    gradients.

    :param rot_tensor: XYZ Euler rotation vector of shape (3,1)
    :param t_tensor: XYZ translation vector of shaape (3,1)
    :return:
    """
    rotmat = roma.rotvec_to_rotmat(rot_tensor)
    rot_opt = torch.stack([
        torch.cat([rotmat[0], torch.zeros(1, device='cuda:1')]),
        torch.cat([rotmat[1], torch.zeros(1, device='cuda:1')]),
        torch.cat([rotmat[2], torch.zeros(1, device='cuda:1')]),
        torch.cat([torch.zeros(3, device='cuda:1'), torch.ones(1, device='cuda:1')])]).reshape(4, 4)
    t_opt = torch.stack([
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device='cuda:1'),
        torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32, device='cuda:1'),
        torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device='cuda:1'),
        torch.cat([t_tensor, torch.ones(1, dtype=torch.float32, device='cuda:1')])]).reshape(4, 4).t()
    return rot_opt, t_opt

# -------------------------------------------------------------------------------------------------


def fit_cube(max_iter          = 5000,
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
    """
    Fit the geometry of a cube of known dimensions and appearance to reference photos
    in order to validate camera parameters.

    :param max_iter:
    :param resolution:
    :param log_interval:
    :param display_interval:
    :param display_res:
    :param out_dir:
    :param log_fn:
    :param basemeshpath:
    :param imdir:
    :param calibs:
    :param texpath:
    :return:
    """
    mp4save_interval = 3
    cams = os.listdir(imdir)
    n_frames = assert_num_frames(cams, imdir)

    # Set up logging.
    if out_dir:
        out_dir = f'{out_dir}/cube_{resolution}'
        print (f'Saving results under {out_dir}')
        writer = imageio.get_writer(f'{out_dir}/progress.mp4', mode='I', fps=30, codec='libx264', bitrate='16M')
    else:
        out_dir = None
        print ('No output directory specified, not saving log or images')
    gl_avg = []
    log_file = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if log_fn:
            log_file = open(f'{out_dir}/{log_fn}', 'wt')

    # object data
    rubiks = data.MeshData(basemeshpath)
    vtxp = torch.tensor(rubiks.vertices, dtype=torch.float32, device='cuda:1')
    pos_idx = torch.tensor(rubiks.faces, dtype=torch.int32, device='cuda:1')
    uv_idx = torch.tensor(rubiks.fuv, dtype=torch.int32, device='cuda:1')
    uv = torch.tensor(rubiks.uv, dtype=torch.float32, device='cuda:1')
    texture = np.array(Image.open(texpath))/255.0
    tex = torch.tensor(texture, dtype=torch.float32, device='cuda:1')

    # learn euler angles, translation and scale
    rotvec = torch.ones(3, dtype=torch.float32, device='cuda:1', requires_grad=True)
    tvec = torch.ones(3, dtype=torch.float32, device='cuda:1', requires_grad=True)
    scale = torch.tensor(1, dtype=torch.float32, device='cuda:1', requires_grad=True)

    # context
    glctx = dr.RasterizeGLContext()
    optimizer = torch.optim.Adam([rotvec, tvec, scale], lr=1e-1)
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
            ref = torch.from_numpy(np.flip(img, 0).copy()).cuda(device='cuda:1')
            # ref = torch.from_numpy(img).cuda(device='cuda:1')

            # lens distortion handled as preprocess in reference images
            projection = torch.tensor(camera.intrinsic_to_projection(intr), dtype=torch.float32, device='cuda:1')
            # projection = torch.tensor(camera.default_projection(), dtype=torch.float32, device='cuda:1')
            modelview = torch.tensor(camera.extrinsic_to_modelview(rot, trans), dtype=torch.float32, device='cuda:1')

            for it in range(max_iter + 1):
                # rotation/translation matrix for offsetting object so that it matches the image
                # camera distortion is handled as preprocessing step on the reference images

                # split [n_vertices * 3] to [n_vertices, 3] as a view of the original tensor
                vtxp_split = torch.reshape(vtxp, (vtxp.shape[0] // 3, 3))
                # vtxp_opt_split = torch.reshape(vtxp_opt, (vtxp_opt.shape[0]//3, 3))

                rot_opt, t_opt = rot_trans_matrices(rotvec, tvec)
                rt = torch.matmul(t_opt, rot_opt)
                mv = torch.matmul(modelview, rt)
                mvp = torch.matmul(projection, mv)
                # vtxp_opt = torch.mul(vtxp, scale)
                mvp_ref = torch.matmul(projection, modelview)

                # render
                colour = render(glctx, mvp, vtxp_split, pos_idx, uv, uv_idx, tex, resolution)

                # Compute loss and train.
                loss = torch.mean((ref - colour*255) ** 2)  # L2 pixel loss.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Print/save log.
                if log_interval and (it % log_interval == 0):
                    gl_val = np.mean(np.asarray(gl_avg))
                    gl_avg = []
                    s = ""
                    s += f"iter={it}, err={loss}"
                    print(s)
                    if log_file:
                        log_file.write(s + "\n")

                display_image = display_interval and (it % display_interval == 0)
                # save_mp4 = mp4save_interval and (it % mp4save_interval == 0)
                if display_image:#  or save_mp4:
                    img_ref = ref.cpu().numpy()
                    img_ref = np.array(img_ref.copy(), dtype=np.float32)/255
                    img_col = np.flip(colour.cpu().detach().numpy(), 0)
                    # img_col = colour.cpu().detach().numpy()
                    result_image = make_img(np.stack([img_ref, img_col]))
                    utils.display_image(result_image, size=display_res)
                    # if save_mp4:
                        # writer.append_data(np.clip(np.rint(img_col * 255.0), 0, 255).astype(np.uint8))

    if writer is not None:
        writer.close()
    if log_file:
        log_file.close()


# -------------------------------------------------------------------------------------------------

def main():
    # Get camera calibration
    # ...
    path = r"C:\Users\Henrik\fpc-diffrend\calibration\2021-07-01\DI_calibration.json"
    with open(path) as json_file:
        calibs = json.load(json_file)

    # Run
    fit_cube(
        max_iter=2000,
        resolution=(1600, 1200),
        log_interval=20,
        display_interval=5,
        display_res=1024,
        out_dir=r"C:\Users\Henrik\fpc-diffrend\data\cube\out",
        log_fn='log.txt',
        basemeshpath=r"C:\Users\Henrik\fpc-diffrend\data\cube\rubiks_bl.obj",
        imdir=r"C:\Users\Henrik\fpc-diffrend\data\cube\20220310\neutrals\rubik_cube\neutral\take0001\fullres",
        calibs=calibs,
        texpath=r"C:\Users\Henrik\fpc-diffrend\data\cube\rubiks.png"
    )

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
