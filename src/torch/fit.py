# builtin
import os
import json

# 3rd party
import numpy as np
import torch
import nvdiffrast.torch as dr
from PIL import Image
import roma as roma
import imageio

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


def blend(v_base, m1, m2, m3, frames):
    """
    Get blended vertex positions from decomposed matrix of vertex positions

    :param v_base: Base mesh vertex positions of shape [3*v], (x,y,z,x,...)
    :param m1: Mapping from frames to learned vertex deltas
    :param m2: Mapping from frames to learned vertex deltas
    :param m3: Learned basis of vertex deltas
    :param frames: One-hot vector of frame index, with value at frame i being 1.
    :return: Blended mesh vertex positions of shape [3*v], (x,y,z,x,...)
    """

    mapped = torch.matmul(m1, frames)
    basis = torch.matmul(m2, mapped)
    prod = torch.matmul(m3, basis)
    vtx_pos = torch.add(v_base, prod)
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
    pos_clip = camera.transform_clip(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=(resolution[0], resolution[1]))

    if enable_mip:
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        colour = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        colour = dr.texture(tex[None, ...], texc, filter_mode='linear')

    colour = dr.antialias(colour, rast_out, pos_clip, pos_idx)
    # colour = colour * torch.clamp(rast_out[..., -1:], 0, 1)
    colour = torch.where(rast_out[..., 3:] > 0, colour, torch.tensor(45.0 / 255.0).cuda())
    return colour[0]

# -------------------------------------------------------------------------------------------------


def setup_dataset(n_frames, n_vertices_x3):
    """
    Set up dataset of vertex matrix decomposition with v_i = v_base + m3*m2*m1*frames.

    m3 can be seen as the learned basis for mesh deltas, m1 and m2 as mapping from frame index to this
    basis. Finding the geometry in optimization thus corresponds to finding values for m1, m2 and m3.

    :return:
    """

    m1 = torch.eye(n_frames, dtype=torch.float32, device='cuda')
    m2 = torch.eye(n_frames, dtype=torch.float32, device='cuda')
    m3 = torch.zeros((n_vertices_x3, n_frames), dtype=torch.float32, device='cuda')

    return m1, m2, m3, torch.zeros(n_frames, dtype=torch.float32, device='cuda')

# -------------------------------------------------------------------------------------------------


def fitTake(max_iter, lr_base, lr_ramp, basemeshpath, localblpath, globalblpath, display_interval,
            log_interval, imdir, calibpath, enable_mip, max_mip_level, texshape, out_dir, resolution,
            mp4_interval, texpath=""):
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
    :param log_interval: Epoch interval for logging loss
    :param imdir: Image directory to take with structure take/camera/frame
    :param calibpath: Path to camera calibration file
    :param texpath: Path to initial texture for face
    :param enable_mip: Boolean whether to enable mipmapping
    :param max_mip_level: Limits the number of mipmaps constructed and used in mipmap-based filter modes
    :param texshape: Shape of the texture with resolution and channels (height, width, channels)
    :param out_dir: Directory to save result data to
    :param resolution: Resolution to render in (height, width)
    :param mp4_interval: Interval in which to save mp4 frames. 0 for no mp4 saving.
    :return:
    """

    if mp4_interval:
        writer = imageio.get_writer(f'{out_dir}/progress.mp4', mode='I', fps=30, codec='libx264', bitrate='16M')
    else:
        writer = None

    flip_opt_interval = 500

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
    uv_idx = torch.tensor(basemesh.fuv, dtype=torch.int32, device='cuda')
    if texpath:
        tex = np.array(Image.open(texpath))/255.0
        tex = tex[..., np.newaxis]
        tex = np.flip(tex, 0)
    else:
        tex = np.random.uniform(low=0.0, high=1.0, size=texshape)
    tex_opt = torch.tensor(tex.copy(), dtype=torch.float32, device='cuda', requires_grad=False)
    t_opt = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda', requires_grad=True)
    # we can't init the rotation to exactly 0.0 as the gradients are then not stable
    rotvec_opt = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32, device='cuda', requires_grad=True)

    # blendshapes and mappings
    n_vertices_x3 = v_base.shape[0]
    m1, m2, m3, v_f = setup_dataset(n_frames, n_vertices_x3)

    # context and optimizer
    print("Setting up RasterizeGLContext and optimizer...")
    glctx = dr.RasterizeGLContext()

    # ================================================================
    # UPDATE PARAMETERS HERE
    optimizer = torch.optim.Adam([m1, m2, m3], lr=lr_base, weight_decay=10e-1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_ramp**(float(x)/float(max_iter)))
    # ================================================================

    # starting camera iteration
    for cam in cams:
        # get camera calibration
        if "pod2texture" not in cam:
            continue
        calib = calibs[cam.split("_")[1]]
        intr = np.asarray(calib['intrinsic'], dtype=np.float32)
        dist = np.asarray(calib['distortion'], dtype=np.float32)
        rot = np.asarray(calib['rotation'], dtype=np.float32)
        trans = np.asarray(calib['translation'], dtype=np.float32)

        camdir = os.path.join(imdir, cam)
        frames = os.listdir(camdir)
        for i, frame in enumerate(frames):
            # reference image to render against
            img = np.array(Image.open(os.path.join(camdir, frame)))
            # img = cv2.undistort(img, intr, dist)
            ref = torch.from_numpy(np.flip(img, 0).copy()).cuda()
            ref = ref.reshape((ref.shape[0], ref.shape[1], 1))

            # set one-hot frame index
            framenum = int(os.path.splitext(frame)[0].split("_")[-1])
            v_f[framenum] = 1.0

            # modelview and projection
            # TODO: how to incorporate camera distortion parameters in projection? in shaders?
            # lens distortion currently handled as preprocess in reference images
            projection = camera.intrinsic_to_projection(intr)
            proj = torch.from_numpy(projection).cuda()
            modelview = camera.extrinsic_to_modelview(rot, trans)
            trans = torch.tensor(camera.translate(0.0, 0.0, 0.0), dtype=torch.float32, device='cuda')
            t_mv = torch.matmul(torch.from_numpy(modelview).cuda(), trans)

            # render
            for it in range(max_iter + 1):
                if it < 500:
                    rigid_trans = camera.rigid_grad(t_opt*0.1, roma.rotvec_to_rotmat(rotvec_opt*0.1))
                else:
                    rigid_trans = camera.rigid_grad(t_opt * 0.05, roma.rotvec_to_rotmat(rotvec_opt * 0.05))
                # print(f"t_opt: {t_opt} --- rotvec_opt: {rotvec_opt} --- rigid_trans: {rigid_trans}")
                tr = torch.matmul(rigid_trans, t_mv)
                mvp = torch.matmul(proj, tr)

                # get blended vertex positions according to eq.
                vtx_pos = blend(v_base, m1, m2, m3, v_f)
                # split [n_vertices * 3] to [n_vertices, 3] as a view of the original tensor
                vtx_pos_split = torch.reshape(vtx_pos, (vtx_pos.shape[0] // 3, 3))

                # render
                colour = render(glctx, mvp, vtx_pos_split, pos_idx, uv, uv_idx, tex_opt, resolution, enable_mip, max_mip_level)

                # if we're optimizing pose, add blur for more tractable optimization landscape
                # built-in gaussian not available for torch tensors since we can't use the right torch3d version
                """if t_opt.requires_grad:
                    blur = utils.GaussianBlur(kernel_size=11)
                    ref_blur = blur(ref)
                    colour_blur = blur(colour)"""

                # Compute loss and train.
                # TODO: add activation constraints (L1 sparsity)
                """if t_opt.requires_grad:
                    loss = torch.mean((ref_blur - colour_blur*255) ** 2)  # L2 pixel loss, *255 to channels from opengl
                else:
                    loss = torch.mean((ref - colour*255) ** 2)  # L2 pixel loss, *255 to channels from opengl"""
                loss = torch.mean((ref - colour*255) ** 2)  # L2 pixel loss, *255 to channels from opengl
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Print loss logging
                log = (log_interval and (it % log_interval == 0))
                if log:
                    print(f"It[{it}] - Loss: {loss} - pos_vtx[1]: {vtx_pos_split[0]} - avg_act: {torch.mean(maps['local'][framenum])}"
                          f" - max_act: {torch.max(maps['local'][framenum])}")

                # change the target of optimization
                if it % flip_opt_interval == 0:
                    # tex_opt.requires_grad = not tex_opt.requires_grad
                    if it == 1000:
                        print("Switched to optimizing blendshape activations!")
                        maps['local'].requires_grad = True
                        t_opt.requires_grad = not t_opt.requires_grad
                        rotvec_opt.requires_grad = not rotvec_opt.requires_grad
                    elif it >= 2000:
                        tex_opt.requires_grad = True
                # Show/save image.
                display_image = (display_interval and (it % display_interval == 0)) or it == max_iter
                save_mp4 = (mp4_interval and (it % mp4_interval == 0))
                if display_image or save_mp4:
                    img_ref = ref.cpu().numpy()
                    img_ref = np.flip(np.array(img_ref.copy(), dtype=np.float32) / 255, 0)
                    img_col = np.flip(colour.cpu().detach().numpy(), 0)
                    result_image = utils.make_img(np.stack([img_ref, img_col]))
                    if display_image:
                        utils.display_image(result_image)
                    if save_mp4:
                        writer.append_data(np.clip(np.rint(result_image * 255.0), 0, 255).astype(np.uint8))
                    # img_out = colour[0].cpu().numpy()
                    # utils.display_image(img_out, size=img.shape[::-1])
            # utils.save_image(os.path.join(out_dir, frame), img_col)
            v_f[framenum] = 0.0
            break
        break

    if writer is not None:
        writer.close()
