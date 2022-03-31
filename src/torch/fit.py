# builtin
import os
import json

# 3rd party
import numpy as np
import torch
import nvdiffrast.torch as dr
from PIL import Image

# local
import src.torch.data as data
import src.torch.utils as utils
import src.torch.camera as camera

# -------------------------------------------------------------------------------------------------


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

# -------------------------------------------------------------------------------------------------


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

# -------------------------------------------------------------------------------------------------


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
    colour = colour * torch.clamp(rast_out[..., -1:], 0, 1)
    return colour

# -------------------------------------------------------------------------------------------------

def setup_dataset(localblpath, globalblpath, n_frames, n_vertices_x3):
    """
    Set up dataset of blendshapes/ml dataset frames and corresponding mappings

    local num_localbls*3v
    global num_globalbls*3v

    :return:
    """
    datasets = {}
    maps = {}
    if globalblpath != "":
        raise Exception("Blending global blendshapes from ml dataset caches not yet implemented")
    if localblpath != "":
        # get data
        objs = os.listdir(localblpath)
        n_meshes = len(objs)
        localbl = np.empty((n_meshes, n_vertices_x3), dtype=np.float32)
        for i, obj in enumerate(objs):
            meshdata = data.MeshData(os.path.join(localblpath, obj))
            localbl[i] = meshdata.vertices

        # shapes
        m_3vb = torch.tensor(localbl, dtype=torch.float32, device='cuda')
        datasets['local'] = m_3vb

        # mappings
        m_bf_face = torch.tensor(np.empty((n_frames, n_meshes)), dtype=torch.float32,
                                 device='cuda', requires_grad=True)
        maps['local'] = m_bf_face

    return datasets, maps, torch.zeros(n_frames)

# -------------------------------------------------------------------------------------------------


def fitTake(max_iter, lr_base, lr_ramp, basemeshpath, localblpath, globalblpath, display_interval,
            imdir, calibpath, enable_mip, max_mip_level, texshape, out_dir):
    """
    Fit one take (continuous range of frames).

    :param max_iter: Max iterations (int)
    :param lr_base: Base learning rate
    :param lr_ramp: Learning rate ramp-down for use with torch.optim.lr_scheduler:
                    lr_base * lr_ramp ^ (epoch/max_iter)
    :param basemeshpath: Path to the base mesh
    :param localblpath: Path to directory of local blendshapes
    :param globalblpath: Path to directory of global blendshapes
    :param display_interval: Epoch interval for displaying render previews
    :param imdir: Image directory to take with structure take/camera/frame
    :param calibs: Camera calibration dict from calibration file for take in question
    :param enable_mip: Boolean whether to enable mipmapping
    :param max_mip_level: Limits the number of mipmaps constructed and used in mipmap-based filter modes
    :param texshape: Shape of the texture with resolution and channels (height, width, channels)
    :param out_dir: Directory to save result data to
    :return:
    """

    cams = os.listdir(imdir)
    n_frames = assertNumFrames(cams, imdir)
    # calibrations
    path = r"C:\Users\Henkka\Projects\invrend-fpc\data\calibration\2021-07-01\DI_calibration.json"
    with open(calibpath) as json_file:
        calibs = json.load(json_file)

    # initialize tensors
    # basemesh
    basemesh = data.MeshData(basemeshpath)
    v_base = torch.tensor(basemesh.vertices, dtype=torch.float32, device='cuda')
    pos_idx = torch.tensor(basemesh.faces, dtype=torch.int32, device='cuda')
    uv = torch.tensor(basemesh.uv, dtype=torch.float32, device='cuda')
    uv_idx = torch.tensor(basemesh.fuv, dtype=torch.float32, device='cuda')
    tex = np.random.uniform(low=0.0, high=255.0, size=texshape)
    tex_opt = torch.tensor(tex, dtype=torch.float32, device='cuda', requires_grad=True)

    # blendshapes and mappings
    n_vertices_x3 = v_base.shape[0]
    datasets, maps, frames = setup_dataset(localblpath, globalblpath, n_frames, n_vertices_x3)

    # context and optimizer
    glctx = dr.RasterizeGLContext()
    optimizer = torch.optim.Adam([maps['local'], tex], lr=lr_base)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_ramp**(float(x)/float(max_iter)))

    for cam in cams:
        # get camera calibration
        if "pod2texture" not in cam:
            continue
        calib = calibs[cam.split("_")[1]]
        intr = np.asarray(calib['intrinsic'], dtype=np.float32)
        # dist = np.asarray(calib['distortion'], dtype=np.float32)
        rot = np.asarray(calib['rotation'], dtype=np.float32)
        trans = np.asarray(calib['translation'], dtype=np.float32)

        camdir = os.path.join(imdir, cam)
        frames = os.listdir(camdir)
        for i, frame in enumerate(frames):
            # reference image to render against
            img = np.array(Image.open(os.path.join(camdir, frame)))
            ref = torch.from_numpy(img).cuda()

            # set one-hot frame index
            framenum = int(os.path.splitext(frame)[0].split("_")[-1])
            frames[framenum] = 1

            # modelview and projection
            # TODO: how to incorporate camera distortion parameters in projection? in shaders?
            # lens distortion currently handled as preprocess in reference images
            projection = camera.intrinsic_to_projection(intr)
            modelview = camera.extrinsic_to_modelview(rot, trans)
            mvp = np.matmul(projection, modelview)

            # render
            for it in range(max_iter + 1):
                # get blended vertex positions according to eq.
                vtx_pos = blend(v_base, maps, datasets, frames)
                # split [n_vertices * 3] to [n_vertices, 3] as a view of the original tensor
                vtx_pos_split = torch.reshape(vtx_pos, (vtx_pos.shape[0] // 3, 3))

                # render
                colour = render(glctx, mvp, vtx_pos_split, pos_idx, uv, uv_idx, tex, img.shape[::-1], enable_mip, max_mip_level)

                # Compute loss and train.
                # TODO: add activation constraints (L1 sparsity)
                loss = torch.mean((ref - colour*255) ** 2)  # L2 pixel loss, *255 to channels from opengl
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Show/save image.
                display_image = (display_interval and (it % display_interval == 0)) or it == max_iter
                if display_image:
                    img_ref = ref.cpu().numpy()
                    img_ref = np.array(img_ref.copy(), dtype=np.float32) / 255
                    img_col = np.flip(colour.cpu().detach().numpy(), 0)
                    result_image = utils.make_img(np.stack([img_ref, img_col]))
                    utils.display_image(result_image)

                    img_out = colour[0].cpu().numpy()
                    utils.display_image(img_out, size=img.shape[::-1])
            utils.save_image(os.path.join(out_dir, frame), img_col)
            frames[framenum] = 0
