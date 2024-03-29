# builtin
import os
import json
import random
import codecs

# 3rd party
import numpy as np
from PIL import Image
import roma as roma
import imageio
import torch
import nvdiffrast.torch as dr
import torchvision.utils
import torchvision.transforms as transforms
import pytorch3d.structures.meshes as meshes
import pytorch3d.loss.mesh_laplacian_smoothing as laplacian
import pytorch3d.loss.mesh_edge_loss as mel
import pytorch3d.loss.mesh_normal_consistency as mnc

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
    assert not any([x != n_frames[0] for x in n_frames]), "All cameras do not have the same number of frames!"
    return (n_frames[0], 2) if n_frames[0] < 100 else (n_frames[0], 3)

# -------------------------------------------------------------------------------------------------

def blend_free(v_base, m1, m2, m3, frames):
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

def blend_combined(v_base, m1, m2, m3, maps, maps_intermediate, datasets, frames, learned_coefficient=1.0):
    """
        Get blended vertex positions from decomposed matrix of vertex positions combined with rig prior

        :param v_base: Base mesh vertex positions of shape [3*v], (x,y,z,x,...)
        :param m1: Mapping from frames to learned vertex deltas
        :param m2: Mapping from frames to learned vertex deltas
        :param m3: Learned basis of vertex deltas

        :param maps: Dict of mappings, one mapping for each dataset: local, global, pca. Shape (n_frames, n_frames)
        :param maps_intermediate: Dict of intermediate mappings, similar as param maps but of shape (n_blendshapes, n_frames)
        :param dataset: Dict of datasets (blendshapes): local, global, pca. Shape (3*n_vertices, n_blendshapes)

        :param frames: One-hot vector of frame index, with value at frame i being 1.
        :param learned_coefficient: Coefficient for weight of learned shape basis

        :return: Blended mesh vertex positions of shape [3*v], (x,y,z,x,...)

    """

    # Decompose vertex positions into a matrix product to make optimization landscape more tractable.
    # Hence, matrices mapped and mapped_intermediate for feeding one-hot frame tensor into blendshape dataset
    mapped = torch.matmul(maps['local'], frames)
    mapped_intermediate = torch.matmul(maps_intermediate['local'], mapped)
    bl_res = torch.matmul(datasets['local'], mapped_intermediate)

    # learned basis and mapping matrices from Laine et al.
    mapped = torch.matmul(m1, frames)
    basis = torch.matmul(m2, mapped)
    prod = torch.matmul(m3, basis)

    vtx_pos = torch.add(v_base, bl_res)
    vtx_pos = torch.add(vtx_pos, learned_coefficient*prod)
    return vtx_pos

# -------------------------------------------------------------------------------------------------

def blend(v_base, maps, maps_intermediate, dataset, frames):
    """
    Get blended vertex positions as in rig eq.

    :param v_base: Base mesh vertex positions of shape [3*v], (x,y,z,x,...)
    :param maps: Dict of mappings, one mapping for each dataset: local, global, pca. Shape (n_frames, n_frames)
    :param maps_intermediate: Dict of intermediate mappings, similar as param maps but of shape (n_blendshapes, n_frames)
    :param dataset: Dict of datasets (blendshapes): local, global, pca. Shape (3*n_vertices, n_blendshapes)
    :param frames: One-hot vector of frame index, with value at frame i being 1.
    :return: Blended mesh vertex positions of shape [3*v], (x,y,z,x,...)
    """

    if 'global' not in dataset:
        # Decompose vertex positions into a matrix product to make optimization landscape more tractable.
        # Hence, matrices mapped and mapped_intermediate for feeding one-hot frame tensor into blendshape dataset
        mapped = torch.matmul(maps['local'], frames)
        mapped_intermediate = torch.matmul(maps_intermediate['local'], mapped)
        bl_res = torch.matmul(dataset['local'], mapped_intermediate)
        vtx_pos = torch.add(v_base, bl_res)
        return vtx_pos
    else:
        mapped = torch.matmul(maps['local'], frames)
        bl_res = torch.matmul(dataset['local'], mapped)

        # blend with ml cache expressions through vertex masks on mouth and eyes
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
    colour = torch.where(rast_out[..., 3:] > 0, colour, torch.tensor(45.0 / 255.0).cuda(device='cuda'))
    return colour[0]

# -------------------------------------------------------------------------------------------------

def setup_dataset_free(n_frames, n_vertices_x3):
    """
    Set up dataset of vertex matrix decomposition with v_i = v_base + m3*m2*m1*frames.
    m3 can be seen as the learned basis for mesh deltas, m1 and m2 as mapping from frame index to this
    basis. Finding the geometry in optimization thus corresponds to finding values for m1, m2 and m3.
    :return:
    """

    m1 = torch.eye(n_frames, dtype=torch.float32, device='cuda', requires_grad=False)
    m2 = torch.eye(n_frames, dtype=torch.float32, device='cuda', requires_grad=False)
    m3 = torch.zeros((n_vertices_x3, n_frames), dtype=torch.float32, device='cuda', requires_grad=False)

    return m1, m2, m3, torch.zeros(n_frames, dtype=torch.float32, device='cuda')


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
    maps_intermediate = {}

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

        # mappings m1
        m_bf_face = torch.zeros(n_frames, n_frames, dtype=torch.float32, device='cuda', requires_grad=False)
        maps['local'] = m_bf_face

        # mappings m2
        m_bf_face_intermediate = torch.eye(n_meshes, n_frames, dtype=torch.float32, device='cuda', requires_grad=False)
        maps_intermediate['local'] = m_bf_face_intermediate

    return datasets, maps, maps_intermediate, torch.zeros(n_frames, dtype=torch.float32, device='cuda')

# -------------------------------------------------------------------------------------------------


def save(meshes, uv, pos_idx, texture, translation, rotation, directory):
    """
    Save the sequence of optimized meshes per frame.

    :param meshes: Tensor of per-frame final meshes of shape (n_frames, n_vertices * 3)
    :param uv: Tensor of original UV coordinates
    :param pos_idx: Tensor of mesh faces (triangles)
    :param texture: Numpy array containing texture
    :param translation: Tensor of shape (n_frames, 3) containing per-frame head xyz translation
    :param rotation: Tensor of shape (n_frames, 4) containing per-frame head translations as quaternions
    :param directory: destination path to save to (str)

    :return:
    """
    print(f"Saving {meshes.shape[0]} meshes...")
    directory = os.path.join(directory, "result")
    if not os.path.isdir(directory):
        os.mkdir(directory)
    try:
        with open(os.path.join(directory, "faces.txt"), mode="r") as f:
            faces = f.readlines()
    except Exception:
        print("faces.txt not found!")
        faces = ""
    for i, mesh in enumerate(meshes):
        with open(os.path.join(directory, f"{str(i)}.obj"), mode="w") as f:
            v = 0
            while v < mesh.shape[0]:
                f.write(f"v {mesh[v]} {mesh[v+1]} {mesh[v+2]}\n")
                v += 3
            for u in uv:
                f.write(f"vt {u[0]} {u[1]}\n")
            f.writelines(faces)

    print(f"Saving texture... with shape {texture.shape}")
    try:
        imageio.imwrite(os.path.join(directory, "texture.png"), (np.flip(texture, 0)*255).astype(np.uint8), format="png")
    except Exception as e:
        print(f"imageio failed with '{str(e)}'")

    print(f"Saving head translation and rotation...")
    try:
        t = translation.cpu().detach().tolist()
        r = rotation.cpu().detach().tolist()
        dictobj = {'translation': t, 'rotation': r}
        json.dump(dictobj, codecs.open(os.path.join(directory, 'pose.json'), 'w', encoding='utf-8'),
                  separators=(',', ':'),
                  sort_keys=True,
                  indent=4)
    except Exception as e:
        print(str(e))
    print("Everything saved successfully.")


# -------------------------------------------------------------------------------------------------


def get_vertex_differentials(vtx_pos, vtx_neigh, n_vertices):
    """
    Compute and return tensor of one-ring neighbour vertex differentials from tensor of vertex positions.
    :param vtx_pos: tensor of vertex positions of shape (n_vertices*3) (x,y,z,x,...)
    :param vtx_neigh: dict of one-ring vertex neighbours by index
    :param n_vertices: number of vertices in mesh
    :return: tensor of vertex differentials
    """
    diffs = torch.zeros(n_vertices, dtype=torch.float32, device='cuda')
    for idx in range(n_vertices):
        # one-vertex differential
        diffs[idx] = vtx_pos[idx] - torch.mean(data.get_vertex_coordinates(vtx_pos, vtx_neigh[idx]))
    return diffs

# -------------------------------------------------------------------------------------------------


def laplacian_regularization(base_vtx_differential, vtx_pos, vertex_neighbours, n_vertices):
    """
    Compute the mesh laplacian regularization term for penalizing local curvature changes.
    :param base_vtx_differential: Uniformly-weighted vertex differentials for the base mesh
    :param vtx_pos: tensor of shape (n_vertices*3) (x,y,z,x,...)
    :param vertex_neighbours: dict of vertex neighbours by vertex number
    :return: loss
    """
    vtx_pos_differential = get_vertex_differentials(vtx_pos, vertex_neighbours, n_vertices)
    loss = torch.mean((base_vtx_differential - vtx_pos_differential)**2)
    return loss

# -------------------------------------------------------------------------------------------------

def fitTake(max_iter,
            lr_base,
            lr_tex_coef,
            lr_ramp,
            lr_t,
            lr_q,
            basemeshpath,
            localblpath,
            globalblpath,
            display_interval,
            log_interval,
            imdir,
            calibpath,
            enable_mip,
            max_mip_level,
            texshape,
            out_dir,
            resolution,
            mp4_interval,
            tex_startlearnratio,
            tex_ramplearnratio,
            free_startlearnratio,
            weight_laplacian,
            weight_meshedge,
            meshedge_target,
            weight_normalconsistency,
            cam_idxs,
            whiten_mean=50,
            whiten_std=25,
            texpath="",
            maskpath="",
            mode="",
            combined_corrective_coefficient=1.0,
            regularize_correctives=False,
            regularize_prior=False):
    """
    Main loop.
    Fit one take (continuous range of frames).

    :param max_iter: Max iterations (int)
    :param lr_base: Base learning rate
    :param lr_tex_coef: Coefficient with which to multiply the texture learning rate w.r.t. lr_base
    :param lr_ramp: Learning rate ramp-down for use with torch.optim.lr_scheduler:
                    lr_base * lr_ramp ^ (epoch/max_iter)
    :param lr_t: Base learning rate for translation vectors
    :param lr_q: Base learning rate for rotation quaternions
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
    :param tex_startlearnratio: Inverse of iteration duration ratio for when to start learning texture
    :param tex_ramplearnratio: Inverse of iteration duratio ratio for when to ramp learning texture lr
    :param maskpath: Path to vertex mask directory
    :param weight_laplacian: Weight coefficient in loss function for laplacian
    :param weight_meshedge: Weight coefficient in loss function for mesh edge length normalization
    :param meshedge_target: Target value for mesh edge length normalization
    :param weight_normalconsistency: Weight coefficient in loss function for mesh normal consistency
    :param cam_idxs: List of camera indices (ints) to use
    :param whiten_mean: Mean value to use for reference image whitening
    :param whiten_std: Standard deviation value to use for reference image whitening
    :param mode: str (prior, free, combined) mode string that determines what mesh data to optimize
    :param combined_corrective_coefficient: Float 0.0 <= x <= 1.0, weight coefficient for learned corrective shapes in combined mode
    :param regularize_correctives: Bool whether to regularize learned corrective deformations with L2 loss
    :param regularize_prior: Bool whether to regularize rig prior activations with L2 loss
    :return:
    """

    args = locals()

    validmodes = ['prior', 'free', 'combined']
    if mode not in validmodes:
        print(f"No valid mode ('{mode}') selected from valid configurations ({validmodes})")
        return

    # miscellaneous setup
    if mp4_interval:
        writer = imageio.get_writer(f'{out_dir}/progress.mp4', mode='I', fps=30, codec='libx264', bitrate='16M')
    else:
        writer = None

    try:
        cams = os.listdir(imdir)
        n_frames, digits = assertNumFrames(cams, imdir)

        # calibrations
        with open(calibpath) as json_file:
            calibs = json.load(json_file)

        # initialize tensors
        # basemesh
        basemesh = data.MeshData(basemeshpath)
        v_base = torch.tensor(basemesh.vertices, dtype=torch.float32, device='cuda')
        print(f"vertices = {basemesh.vertices.size}")
        return
        v_base_split = torch.reshape(v_base, (v_base.shape[0] // 3, 3))
        pos_idx = torch.tensor(basemesh.faces, dtype=torch.int32, device='cuda')
        v_base_loss_mesh = meshes.Meshes(verts=[v_base_split], faces=[pos_idx]).cuda()
        uv = torch.tensor(basemesh.uv, dtype=torch.float32, device='cuda')
        uv_idx = torch.tensor(basemesh.fuv, dtype=torch.int32, device='cuda')
        if texpath:
            tex = np.array(Image.open(texpath)) / 255.0
            tex = tex[..., np.newaxis]
            tex = np.flip(tex, 0)
        else:
            tex = np.random.uniform(low=0.0, high=1.0, size=texshape)
        tex_opt = torch.tensor(tex.copy(), dtype=torch.float32, device='cuda', requires_grad=True)

        # per-camera pose optimization
        # t_opt = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda', requires_grad=True)
        t_opt = torch.zeros([9, 3], dtype=torch.float32, device='cuda', requires_grad=True)
        # initial unit quaternion for rotation optimization
        # q_opt = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device='cuda', requires_grad=True)
        q_opt = torch.zeros([9, 4], dtype=torch.float32, device='cuda', requires_grad=False)
        q_opt[:, 3] = 1.0
        q_opt.requires_grad = True
        cam_idx_tensor = torch.zeros(9, dtype=torch.float32, device='cuda', requires_grad=False)

        per_frame_t = torch.zeros([n_frames, 3], dtype=torch.float32, device='cuda', requires_grad=True)
        per_frame_q = torch.zeros([n_frames, 4], dtype=torch.float32, device='cuda', requires_grad=False)
        per_frame_q[:, 3] = 1.0
        per_frame_q.requires_grad = True

        # final shapes
        result = torch.empty(size=(n_frames, basemesh.vertices.shape[0]), dtype=torch.float32, device='cuda')

        # blendshapes and mappings
        n_vertices_x3 = v_base.shape[0]
        datasets, maps, maps_intermediate, v_f = setup_dataset(localblpath, globalblpath, n_frames, n_vertices_x3,
                                                               basemesh.vertices)
        m1, m2, m3, v_f = setup_dataset_free(n_frames, n_vertices_x3)

        corrective_lr = lr_base
        if mode == "prior":
            maps['local'].requires_grad = True
            maps_intermediate['local'].requires_grad = True
            blendFunction = blend
        elif mode == "free":
            m1.requires_grad = True
            m2.requires_grad = True
            m3.requires_grad = True
            blendFunction = blend_free
        elif mode == "combined":
            # start learning corrective shapes after optimizing halfway
            maps['local'].requires_grad = True
            maps_intermediate['local'].requires_grad = True
            blendFunction = blend_combined
            corrective_lr = lr_base * 0.1

        # context and optimizer
        print("Setting up RasterizeGLContext and optimizer...")
        glctx = dr.RasterizeGLContext(device='cuda')
        # ================================================================
        # UPDATE PARAMETERS HERE
        """optimizer = torch.optim.Adam([{"params": maps['local'], 'lr': lr_base, 'weight_decay': 10e-1},
                                      {"params": maps_intermediate['local'], 'lr': lr_base, 'weight_decay': 10e-1 },
                                      {"params": t_opt, 'lr': 10e-4 * 0.1},
                                      {"params": q_opt, 'lr': 10e-4 * 0.1},
                                      {"params": tex_opt, 'lr': 10e-5 * 0.5, 'weight_decay': 0.0}],
                                     lr=lr_base, weight_decay=10e-1)"""
        optimizer = torch.optim.Adam([{"params": m1, 'lr': corrective_lr},
                                      {"params": m2, 'lr': corrective_lr},
                                      {"params": m3, 'lr': corrective_lr},
                                      {"params": maps['local'], 'lr': lr_base},
                                      {"params": maps_intermediate['local'], 'lr': lr_base},
                                      {"params": t_opt, 'lr': lr_t},
                                      {"params": q_opt, 'lr': lr_q},
                                      {"params": per_frame_t, 'lr': lr_t},
                                      {"params": per_frame_q, 'lr': lr_q},
                                      {"params": tex_opt, 'lr': lr_base * lr_tex_coef}], lr=lr_base)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lambda x: lr_ramp ** (
                                                              float(x) / float(max_iter)))
        blurrer = transforms.GaussianBlur(kernel_size=(31, 31))
        # ================================================================

        # local response norm for image contrast normalization
        lrn = torch.nn.LocalResponseNorm(2)
        # lrn3 = torch.nn.LocalResponseNorm(3)

        # set up camera data for lookup
        calib_lookup = []
        for c, cam in enumerate(cams):
            calib = calibs[cam.split("_")[1]]
            intr = np.asarray(calib['intrinsic'], dtype=np.float32)
            dist = np.asarray(calib['distortion'], dtype=np.float32)
            rot = np.asarray(calib['rotation'], dtype=np.float32)
            trans_calib = np.asarray(calib['translation'], dtype=np.float32)
            calib_lookup.append({'cam': cam, 'intr': intr, 'dist': dist, 'rot': rot, 'trans_calib': trans_calib})

        # start camera & frame iteration
        for i in range(max_iter):
            cam_idx = random.choice(cam_idxs)
            frame_idx = random.randint(0, n_frames-1)

            # reference image to render against
            camdir = os.path.join(imdir, calib_lookup[cam_idx]['cam'])
            img = np.array(Image.open(os.path.join(camdir, f"{calib_lookup[cam_idx]['cam']}_{frame_idx:0{digits}d}.tif")))
            img = np.clip(img, 0, 140)
            ref = torch.tensor(np.flip(img, 0,).copy(), dtype=torch.float32, device='cuda')
            ref = ref.reshape((ref.shape[0], ref.shape[1], 1))

            # set one-hot frame- and camera indices
            v_f[frame_idx] = 1.0
            cam_idx_tensor[cam_idx] = 1.0

            # modelview and projection
            # lens distortion currently handled as preprocess in reference images
            projection = camera.intrinsic_to_projection(calib_lookup[cam_idx]['intr'])
            proj = torch.from_numpy(projection).cuda(device='cuda')
            modelview = camera.extrinsic_to_modelview(calib_lookup[cam_idx]['rot'],
                                                      calib_lookup[cam_idx]['trans_calib'])
            trans = torch.tensor(camera.translate(0.0, 170.0, 0.0), dtype=torch.float32, device='cuda')
            t_mv = torch.matmul(torch.from_numpy(modelview).cuda(device='cuda'), trans)
            rigid_trans = camera.rigid_grad(torch.matmul(cam_idx_tensor, t_opt),
                              roma.unitquat_to_rotmat(torch.matmul(cam_idx_tensor, q_opt)))
            rigid_trans_pose = camera.rigid_grad(torch.matmul(v_f, per_frame_t),
                                            roma.unitquat_to_rotmat(torch.matmul(v_f, per_frame_q)))
            tr = torch.matmul(rigid_trans, t_mv)
            tr_pose = torch.matmul(rigid_trans_pose, tr)
            mvp = torch.matmul(proj, tr_pose)

            # get blended vertex positions according to eq.
            if mode == "prior":
                vtx_pos = blend(v_base, maps, maps_intermediate, datasets, v_f)
            elif mode == "free":
                vtx_pos = blend_free(v_base, m1, m2, m3, v_f)
            elif mode == "combined":
                vtx_pos = blend_combined(v_base, m1, m2, m3, maps, maps_intermediate,
                                         datasets, v_f, learned_coefficient=0.5)
            # split [n_vertices * 3] to [n_vertices, 3] as a view of the original tensor
            vtx_pos_split = torch.reshape(vtx_pos, (vtx_pos.shape[0] // 3, 3))

            # render
            colour = render(glctx, mvp, vtx_pos_split, pos_idx, uv, uv_idx, tex_opt, resolution, enable_mip,
                            max_mip_level)

            """
            =======================
            Compute loss and train.
            =======================
            """

            # L2 pixel loss, *255 to channels from opengl. Second loss term to penalize large translations
            # mesh laplacian term through pytorch3d
            loss_mesh = meshes.Meshes(verts=[vtx_pos_split], faces=[pos_idx]).cuda()
            loss = torch.mean((ref - colour*255) ** 2) + \
                    weight_meshedge*mel(loss_mesh, 0.1) + \
                    weight_laplacian*laplacian(loss_mesh)**2 + \
                    weight_normalconsistency*mnc(loss_mesh) #+  \
                    # torch.mean(torch.sum(torch.matmul(m2, m1) ** 2, dim=0))
            if regularize_correctives and mode == "combined" and i > max_iter/2:
                # regularize the learned corrective deformations
                mapped = torch.matmul(m1, v_f)
                basis = torch.matmul(m2, mapped)
                deformations = torch.matmul(m3, basis)
                loss += torch.mean(deformations ** 2)

            # todo: add a loss term to reward pose optimization w.r.t. blendshape activations
            if regularize_prior and mode == "prior":
                mapped = torch.matmul(maps['local'], v_f)
                mapped_intermediate = torch.matmul(maps_intermediate['local'], mapped)
                loss += torch.mean(mapped_intermediate ** 2)

            with torch.no_grad():
                if not i % 500:
                    print(f"=== MEL: {weight_meshedge*mel(loss_mesh, meshedge_target)} ---"
                          f" LAP: {weight_laplacian*laplacian(loss_mesh)**2} ---"
                          f" MNC: {weight_normalconsistency*mnc(loss_mesh)}")

            if mode == "combined":
                # start learning corrective shapes after optimizing halfway
                if i > max_iter/2:
                    m1.requires_grad = True
                    m2.requires_grad = True
                    m3.requires_grad = True

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # scale and normalize quaternionS
            with torch.no_grad():
                q_opt /= torch.sum(q_opt ** 2) ** 0.5
                per_frame_q /= torch.sum(per_frame_q ** 2) ** 0.5

            # Print loss logging
            log = (log_interval and (i % log_interval == 0))
            if log:
                print(f"Img {frame_idx} cam {cam_idx} - It[{i}] - Loss: {loss} - lr: {scheduler.get_lr()}")

            # Show/save image.
            display_image = (display_interval and (i % display_interval == 0)) or i == max_iter
            save_mp4 = (mp4_interval and (i % mp4_interval == 0) and i)
            if display_image or save_mp4:
                # img_ref = torch.reshape(ref_blur, (ref_blur.shape[1], ref_blur.shape[2], ref_blur.shape[0])).cpu().numpy()
                img_ref = ref.cpu().numpy()
                img_ref = np.flip(np.array(img_ref.copy(), dtype=np.float32) / 255, 0)
                # img_col = np.flip(torch.reshape(colour_blur, (colour_blur.shape[1], colour_blur.shape[2], colour_blur.shape[0])).cpu().detach().numpy(), 0)
                img_col = np.flip(colour.cpu().detach().numpy(), 0)
                result_image = utils.make_img(np.stack([img_ref, img_col]))
                if display_image:
                    utils.display_image(result_image)
                if save_mp4:
                    writer.append_data(np.clip(np.rint(result_image * 255.0), 0, 255).astype(np.uint8))

            v_f[frame_idx] = 0.0
            cam_idx_tensor[cam_idx] = 0.0
            result[frame_idx] = vtx_pos.detach().clone()

    except KeyboardInterrupt:
        if writer is not None:
            writer.close()
    else:
        if writer is not None:
            writer.close()

    save(result, uv, pos_idx, tex_opt.cpu().detach().clone().numpy(),
         per_frame_t.cpu(), per_frame_q.cpu(), out_dir)

    # save config file with settings
    with open(os.path.join(out_dir, "config.txt"), 'w') as f:
        for arg in args:
            f.write(f"{arg}: '{args[arg]}'\n")
    print("Done")
