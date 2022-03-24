# builtin
import os

# 3rd party
import numpy as np
import torch
import nvdiffrast.torch as dr
from PIL import Image

# local
import data
import utils
import camera

# -----------------------------------------------------------------


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

# -----------------------------------------------------------------


def blend(v_base, maps, dataset, frames):
    """
    Get blended vertex positions as in rig eq.

    :param v_base: Base mesh vertex positions of shape [3*v], (x,y,z,x,...)
    :param maps: Dict of mappings, one mapping for each dataset: local, global, pca
    :param dataset: Dict of datasets (blendshapes): local, global, pca
    :param frames: One-hot vector of frame index, with value at frame i being 1.
    :return: Blended mesh vertex positions of shape [3*v], (x,y,z,x,...)
    """
    if 'global' not in dataset:
        mapped = torch.matmul(maps['local'], frames)
        bl_res = torch.matmul(dataset['local'], mapped)
        vtx_pos = torch.add(v_base, bl_res)
        return vtx_pos

# -----------------------------------------------------------------


def render(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution, enable_mip, max_mip_level):
    """
    Render object using Nvdiffrast

    :param glctx: OpenGL context of type RasterizeGLContext
    :param mtx: Modelview + projection matrix
    :param pos: Vertex position tensor with dtype torch.float32 with shape [num_vertices, 4]
    :param pos_idx: Triangle tensor with shape [num_triangles, 3]
    :param uv: Tensor containing per-pixel texture coordinates with shape [minibatch_size, height, width, 2]
    :param uv_idx: Triangle tensor from texture coordinate indices with shape [num_triangles, 3] and dtype torch.int32
    :param tex: Texture tensor with dtype torch.float32 with shape [minibatch_size, tex_height, tex_width, tex_channels]
    :param resolution: Image resolution [height, width]
    :param enable_mip: Bool whether to enable mipmapping
    :param max_mip_level: Limits the number of mipmaps constructed and used in mipmap-based filter modes.
    :return:
    """
    pos_clip = camera.transformClip(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])

    if enable_mip:
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        colour = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        colour = dr.texture(tex[None, ...], texc, filter_mode='linear')

    colour = dr.antialias(colour, rast_out, pos_clip, pos_idx)
    # color = color * torch.clamp(rast_out[..., -1:], 0, 1)  # Mask out background?
    return colour

# -----------------------------------------------------------------


def fitTake(max_iter, lr_base, lr_ramp, basemesh, localbl, globalbl, display_interval, imdir, calibs, enable_mip,
            max_mip_level, texshape):
    """
    Fit one take (continuous range of frames).

    :param max_iter: Max iterations (int)
    :param lr_base: Base learning rate
    :param lr_ramp: Learning rate ramp-down for use with torch.optim.lr_scheduler:
                    lr_base * lr_ramp ^ (epoch/max_iter)
    :param basemesh: data.Meshdata from the basemesh
    :param localbl: Matrix of local blendshapes of shape num_localbls*3v
    :param globalbl: Matrix of global blendshapes of shape num_globalbls*3v
    :param display_interval: Epoch interval for displaying render previews
    :param imdir: Image directory to take with structure take/camera/frame
    :param calibs: Camera calibration dict from calibration file for take in question
    :param enable_mip: Boolean whether to enable mipmapping
    :param max_mip_level: Limits the number of mipmaps constructed and used in mipmap-based filter modes
    :param texshape: Shape of the texture with resolution and channels (height, width, channels)
    :return:
    """
    cams = os.listdir(imdir)
    n_frames = assertNumFrames(cams, imdir)

    # initialize tensors
    # basemesh
    v_base = torch.tensor(basemesh.vtx, dtype=torch.float32, device='cuda')
    uv = torch.tensor(basemesh.uv, dtype=torch.float32, device='cuda')
    uv_idx = torch.tensor(basemesh.fuv, dtype=torch.float32, device='cuda')
    tex = np.random.uniform(low=0.0, high=255.0, size=texshape)
    tex_opt = torch.tensor(tex, dtype=torch.float32, device='cuda', requires_grad=True)
    pos_idx = torch.tensor(basemesh.faces, dtype=torch.int32, device='cuda')

    # blendshapes and mappings
    if globalbl:
        raise Exception("Blending global blendshapes from ml dataset caches not yet implemented")
    dataset = {}
    m_3vb = torch.tensor(localbl, dtype=torch.float32, device='cuda')
    dataset['local'] = m_3vb
    m_bf_face = torch.tensor(np.empty((localbl.shape[0], n_frames)), dtype=torch.float32,
                             device='cuda', requires_grad=True)
    maps = {}
    maps['local'] = m_bf_face
    v_f = torch.zeros(n_frames)

    # context and optimizer
    glctx = dr.RasterizeGLContext()
    optimizer = torch.optim.Adam([m_bf_face, tex], lr=lr_base)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_ramp**(float(x)/float(max_iter)))

    for cam in cams:
        # get camera calibration
        calib = calibs[cam.split("_")[1]]
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

            # set one-hot frame index
            framenum = os.path.splitext(frame)[0].split("_")[-1]
            v_f[framenum] = 1

            # modelview and projection
            # TODO: how to incorporate camera distortion parameters in projection? in shaders?
            # lens distortion currently handled as preprocess in reference images
            projection = camera.intrinsicToProjection(intr)
            modelview = camera.extrinsicToModelview(rot, trans)
            mvp = np.matmul(projection, modelview).astype(np.float32)

            # render
            for it in range(max_iter + 1):
                # get blended vertex positions according to eq.
                vtx_pos = blend(v_base, maps, dataset, v_f)
                # split [n_vertices * 3] to [n_vertices, 3] as a view of the original tensor
                vtx_pos_split = torch.reshape(vtx_pos, (vtx_pos.shape[0] // 3, 3))

                # render
                colour = render(glctx, mvp, vtx_pos_split, pos_idx, uv, uv_idx, tex, img.shape[::-1], enable_mip, max_mip_level)

                # Compute loss and train.
                # TODO: add activation constraints (L1 sparsity)
                loss = torch.mean((ref - colour) ** 2)  # L2 pixel loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Show/save image.
                display_image = display_interval and (it % display_interval == 0)
                if display_image:
                    img_out = colour[0].cpu().numpy()
                    utils.display_image(img_out, size=img.shape[::-1])
            v_f[framenum] = 0