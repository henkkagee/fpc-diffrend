# builtin
import os
import json

# 3rd party
import numpy as np
import torch
from torch.nn.functional import conv2d
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
    pos_clip = camera.transform_clip(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=(resolution[0], resolution[1]))

    if enable_mip:
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        colour = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        colour = dr.texture(tex[None, ...], texc, filter_mode='linear')

    colour = dr.antialias(colour, rast_out, pos_clip, pos_idx)
    colour = torch.where(rast_out[..., 3:] > 0, colour, torch.tensor(45.0 / 255.0).cuda())
    return colour[0]

# -------------------------------------------------------------------------------------------------


def setup_dataset(localblpath, globalblpath, n_frames, n_vertices_x3, v_basemesh):
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

        # not using data.MeshData to speed things up as we only need the vertex positions
        print("Collecting blendshapes...")
        for i, obj in enumerate(objs):
            if i % 50 == 0:
                print(f"Blendshape {i}/{n_meshes}")
            vertices = []
            with open(os.path.join(localblpath, obj), 'r') as f:
                # vertex positions x,y,z,x,...,z
                for line in f:
                    if line.startswith("v "):
                        vertices.extend([float(x) for x in line.strip().split(" ")[1:]])
            # per-vertex deltas
            localbl[i] = np.subtract(np.asarray(vertices, dtype=np.float32), v_basemesh)

        # shapes
        m_3vb = torch.tensor(localbl.transpose(), dtype=torch.float32, device='cuda')
        datasets['local'] = m_3vb

        # mappings
        m_bf_face = torch.zeros(n_meshes, n_frames, dtype=torch.float32, device='cuda', requires_grad=True)
        maps['local'] = m_bf_face

    return datasets, maps, torch.zeros(n_frames, dtype=torch.float32, device='cuda')

# -------------------------------------------------------------------------------------------------


def save(meshes, uv, pos_idx, directory):
    """
    Save the sequence of optimized meshes per frame.

    :param meshes: Tensor of per-frame final meshes of shape (n_frames, n_vertices * 3)
    :param uv: Tensor of original UV coordinates
    :param pos_idx: Tensor of mesh faces (triangles)

    :return:
    """
    print(f"Saving {meshes.shape[0]} meshes...")
    directory = os.path.join(directory, "result")
    if not os.path.isdir(directory):
        os.mkdir(directory)
    with open(os.path.join(directory, "faces.txt"), mode="r") as f:
        faces = f.readlines()
    for i, mesh in enumerate(meshes):
        with open(os.path.join(directory, f"{str(i)}.obj"), mode="w") as f:
            v = 0
            while v < mesh.shape[0]:
                f.write(f"v meshes {mesh[v]} {mesh[v+1]} {mesh[v+2]}\n")
                v += 3
            for u in uv:
                f.write(f"vt {u[0]} {u[1]}\n")
            f.writelines(faces)

# -------------------------------------------------------------------------------------------------


def fitTake(max_iter, lr_base, lr_ramp, pose_lr, basemeshpath, localblpath, globalblpath, display_interval,
            log_interval, imdir, calibpath, enable_mip, max_mip_level, texshape, out_dir, resolution,
            mp4_interval, texpath=""):
    """
    Fit one take (continuous range of frames).

    :param max_iter: Max iterations (int)
    :param lr_base: Base learning rate
    :param lr_ramp: Learning rate ramp-down for use with torch.optim.lr_scheduler:
                    lr_base * lr_ramp ^ (epoch/max_iter)
    :param pose_lr: Learning rate for translation and rotation optimization
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

    # miscellaneous setup
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
    tex_opt = torch.tensor(tex.copy(), dtype=torch.float32, device='cuda', requires_grad=True)
    t_opt = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda', requires_grad=True)
    # initial unit quaternion for rotation optimization
    q_opt = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device='cuda', requires_grad=True)
    # final shapes
    result = torch.empty(size=(n_frames, basemesh.vertices.shape[0]), dtype=torch.float32, device='cuda')

    # blendshapes and mappings
    n_vertices_x3 = v_base.shape[0]
    datasets, maps, v_f = setup_dataset(localblpath, globalblpath, n_frames, n_vertices_x3, basemesh.vertices)

    # context and optimizer
    print("Setting up RasterizeGLContext and optimizer...")
    glctx = dr.RasterizeGLContext(device='cuda')

    # ================================================================
    # UPDATE PARAMETERS HERE
    # optimizer = torch.optim.Adam([maps['local'], t_opt, q_opt], lr=lr_base, weight_decay=10e-1)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_ramp ** (float(x) / float(max_iter)))
    # local response norm for image contrast normalization
    lrn = torch.nn.LocalResponseNorm(2)
    # lrn3 = torch.nn.LocalResponseNorm(3)
    # ================================================================

    # starting camera iteration
    for cam in cams:
        # get camera calibration
        calib = calibs[cam.split("_")[1]]
        intr = np.asarray(calib['intrinsic'], dtype=np.float32)
        dist = np.asarray(calib['distortion'], dtype=np.float32)
        rot = np.asarray(calib['rotation'], dtype=np.float32)
        trans_calib = np.asarray(calib['translation'], dtype=np.float32)

        camdir = os.path.join(imdir, cam)
        frames = os.listdir(camdir)
        for i, frame in enumerate(frames):
            optimizer = torch.optim.Adam([maps['local'], t_opt, q_opt], lr=lr_base, weight_decay=10e-1)
            # optimizer = torch.optim.Adam([tex_opt], lr=lr_base, weight_decay=10e-1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_ramp ** (
                    float(x) / float(max_iter)))

            # reference image to render against
            img = np.array(Image.open(os.path.join(camdir, frame)))
            ref = torch.tensor(np.flip(img, 0).copy(), dtype=torch.float32, device='cuda')
            ref = ref.reshape((ref.shape[0], ref.shape[1], 1))
            ref_norm = lrn(ref.permute(0, 2, 1))
            ref_norm = ref_norm.permute(0, 2, 1)
            smoothing = utils.GaussianSmoothing(1, 32, 1)
            smoothing = smoothing.to('cuda')
            ref_blur = smoothing(torch.reshape(ref_norm, (1, ref_norm.shape[2], ref_norm.shape[0], ref_norm.shape[1])))

            # set one-hot frame index
            framenum = int(os.path.splitext(frame)[0].split("_")[-1])
            v_f[framenum] = 1.0

            # modelview and projection
            # lens distortion currently handled as preprocess in reference images
            projection = camera.intrinsic_to_projection(intr)
            proj = torch.from_numpy(projection).cuda()
            modelview = camera.extrinsic_to_modelview(rot, trans_calib)
            trans = torch.tensor(camera.translate(0.0, 0.0, 0.0), dtype=torch.float32, device='cuda')
            t_mv = torch.matmul(torch.from_numpy(modelview).cuda(), trans)

            # render
            for it in range(max_iter + 1):

                if it < 500:
                    rigid_trans = camera.rigid_grad(t_opt * 0.10, roma.unitquat_to_rotmat(q_opt))
                else:
                    rigid_trans = camera.rigid_grad(t_opt * 0.01, roma.unitquat_to_rotmat(q_opt))
                tr = torch.matmul(rigid_trans, t_mv)
                mvp = torch.matmul(proj, tr)


                # get blended vertex positions according to eq.
                vtx_pos = blend(v_base, maps, datasets, v_f)
                # split [n_vertices * 3] to [n_vertices, 3] as a view of the original tensor
                vtx_pos_split = torch.reshape(vtx_pos, (vtx_pos.shape[0] // 3, 3))

                # render
                colour = render(glctx, mvp, vtx_pos_split, pos_idx, uv, uv_idx, tex_opt, resolution, enable_mip, max_mip_level)


                """
                =======================
                Compute loss and train.
                =======================
                """
                # TODO: add activation constraints (L1 sparsity)

                # local contrast (response) normalization over channels to account for
                # lighting changes between reference and rendered image
                # torch local response norm needs channel dimension to be in dim 1
                colour_norm = lrn(colour.permute(0, 2, 1))
                # permute back original shape from lrn shape requirements (need to have channel back in dim 2)
                colour_norm = colour_norm.permute(0, 2, 1)
                # blur before calculating pixel space loss using gaussian kernel of size 32
                # more tractable optimization landscape
                # built-in gaussian not available for torch tensors since we can't use the right torch3d version
                colour_blur = smoothing(torch.reshape(colour_norm, (1, colour_norm.shape[2], colour_norm.shape[0], colour_norm.shape[1])))

                loss = torch.mean((ref_blur - colour_blur*255) ** 2)  # L2 pixel loss, *255 to channels from opengl
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # scale and normalize quaternion
                with torch.no_grad():
                    q_opt /= torch.sum(q_opt**2) ** 0.5

                # Print loss logging
                log = (log_interval and (it % log_interval == 0))
                if log:
                    print(f"It[{it}] - Loss: {loss} - pos_vtx[1]: {vtx_pos_split[0]} - avg_act: {torch.mean(maps['local'][framenum])}"
                          f" - max_act: {torch.max(maps['local'][framenum])}")

                """# change the target of optimization
                if it % flip_opt_interval == 0:
                    # tex_opt.requires_grad = not tex_opt.requires_grad
                    if it == 1000:
                        print("Switched to optimizing blendshape activations!")
                        maps['local'].requires_grad = True
                        t_opt.requires_grad = not t_opt.requires_grad
                        q_opt.requires_grad = not q_opt.requires_grad
                    elif it >= 2000:
                        tex_opt.requires_grad = True"""

                # Show/save image.
                display_image = (display_interval and (it % display_interval == 0)) or it == max_iter
                save_mp4 = (mp4_interval and (it % mp4_interval == 0) and it)
                if display_image or save_mp4:
                    img_ref = torch.reshape(ref_blur, (ref_blur.shape[2], ref_blur.shape[3], ref_blur.shape[1])).cpu().numpy()
                    # img_ref = ref.cpu().numpy()
                    img_ref = np.flip(np.array(img_ref.copy(), dtype=np.float32) / 255, 0)
                    img_col = np.flip(torch.reshape(colour_blur, (colour_blur.shape[2], colour_blur.shape[3], colour_blur.shape[1])).cpu().detach().numpy(), 0)
                    # img_col = np.flip(colour.cpu().detach().numpy(), 0)
                    result_image = utils.make_img(np.stack([img_ref, img_col]))
                    if display_image:
                        utils.display_image(result_image)
                    if save_mp4:
                        writer.append_data(np.clip(np.rint(result_image * 255.0), 0, 255).astype(np.uint8))
                    # img_out = colour[0].cpu().numpy()
                    # utils.display_image(img_out, size=img.shape[::-1])

            # utils.save_image(os.path.join(out_dir, frame), img_col)
            v_f[framenum] = 0.0
            result[framenum] = vtx_pos

    if writer is not None:
        writer.close()

    save(result, uv, pos_idx, out_dir)
    print("Done")
