# builtin
import os
import json
import codecs

# 3rd party
import numpy as np
import torch
import nvdiffrast.torch as dr
from PIL import Image
import imageio
import roma

# local
import src.torch.data as data
import src.torch.utils as utils
import src.torch.camera as camera

# -------------------------------------------------------------------------------------------------

def render(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution):
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

    texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
    colour = dr.texture(tex[None, ...], texc, filter_mode='linear')

    colour = dr.antialias(colour, rast_out, pos_clip, pos_idx)
    colour = torch.where(rast_out[..., 3:] > 0, colour, torch.tensor(45.0 / 255.0).cuda(device='cuda'))
    return colour[0]

# -------------------------------------------------------------------------------------------------

names = ["d120_prior_1200000_hmc_yoffset"]

for optimname in names:
    print(optimname)
    if "yoffset" in optimname:
        offset = 170.0
    else:
        offset = 0
    # optimname = "d120_prior_1200000_yoffset_regularize"
    wireframe = True
    n_frames = 120
    reproduce_pose = False
    write_imgs = False

    # Get objs
    REFDIR = r"W:\thesis\results\reference\d120\pod2colour_pod2primary"
    DIR = r"W:\thesis\results\safe_final\d120_prior_1200000_yoffset\result"
    objs = os.listdir(DIR)

    if wireframe:
        texpath = os.path.join(DIR, "ilkka_villi_anchor_greyscale_fix_eyes_wireframe_transparency_2.png")
    else:
        texpath = os.path.join(DIR, "texture.png")

    # common mesh info
    basemesh = data.MeshData(os.path.join(DIR, "basemesh.obj"))
    pos_idx = torch.tensor(basemesh.faces, dtype=torch.int32, device='cuda')
    uv = torch.tensor(basemesh.uv, dtype=torch.float32, device='cuda')
    uv_idx = torch.tensor(basemesh.fuv, dtype=torch.int32, device='cuda')
    tex = np.array(Image.open(texpath))/255.0
    tex = tex[..., np.newaxis]
    tex = np.flip(tex, 0)
    tex = torch.tensor(tex.copy(), dtype=torch.float32, device='cuda', requires_grad=True)
    resolution = (1600, 1200)

    glctx = dr.RasterizeGLContext(device='cuda')

    # get camera calibration
    calibpath = r"C:\Users\Henrik\fpc-diffrend\calibration\combined\calibration.json"
    cam = "pod2primary"
    with open(calibpath) as json_file:
        calibs = json.load(json_file)
    calib = calibs[cam]
    intr = np.asarray(calib['intrinsic'], dtype=np.float32)
    dist = np.asarray(calib['distortion'], dtype=np.float32)
    rot = np.asarray(calib['rotation'], dtype=np.float32)
    trans_calib = np.asarray(calib['translation'], dtype=np.float32)

    # saved pose tensors
    if reproduce_pose:
        obj_text = codecs.open(os.path.join(DIR, 'pose.json'), 'r', encoding='utf-8').read()
        dictobj = json.loads(obj_text)
        pose_trans = torch.tensor(dictobj['translation'], device='cuda')
        pose_rot = torch.tensor(dictobj['rotation'], device='cuda')

    glctx = dr.RasterizeGLContext(device='cuda')
    writer = imageio.get_writer(f'{DIR}/result_overlay{"_wireframe" if wireframe else ""}{"_pose" if reproduce_pose else ""}_trans.mp4',
                                mode='I', fps=30, codec='libx264', bitrate='16M')

    v_f = torch.zeros(n_frames, dtype=torch.float32, device='cuda')

    for i in range(0, n_frames):

        v_f[i] = 1.0

        vertices = []
        ref = np.array(Image.open(os.path.join(REFDIR, f"pod2colour_pod2primary_{i:03}.tif")))
        ref = ref.reshape((ref.shape[0], ref.shape[1], 1))
        with open(os.path.join(DIR, f"{i}.obj"), 'r') as f:
            for line in f:
                if line.startswith("v "):
                    vertices.extend([float(x) for x in line.strip().split(" ")[1:]])
                elif line.startswith("vt "):
                    break

        vtx_pos = torch.tensor(vertices, dtype=torch.float32, device='cuda')

        projection = camera.intrinsic_to_projection(intr)
        proj = torch.from_numpy(projection).cuda(device='cuda')
        modelview = camera.extrinsic_to_modelview(rot, trans_calib)
        trans = torch.tensor(camera.translate(0.0, offset, 0.0), dtype=torch.float32, device='cuda')
        if reproduce_pose:
            rigid_trans_pose = camera.rigid_grad(torch.matmul(v_f, pose_trans),
                                             roma.unitquat_to_rotmat(torch.matmul(v_f, pose_rot)))
        t_mv = torch.matmul(torch.from_numpy(modelview).cuda(device='cuda'), trans)
        if reproduce_pose:
            t_mv = torch.matmul(rigid_trans_pose, t_mv)
        mvp = torch.matmul(proj, t_mv)

        vtx_pos_split = torch.reshape(vtx_pos, (vtx_pos.shape[0] // 3, 3))
        colour = render(glctx, mvp, vtx_pos_split, pos_idx, uv, uv_idx, tex, resolution) * 255.0

        """img_col = np.flip(torch.reshape(
            colour, (colour.shape[2], colour.shape[3], colour.shape[1])).cpu().detach().numpy(), 0)"""

        img_col = np.flip(colour.cpu().detach().numpy())
        img_col = np.flip(img_col, 1)

        # result_image = utils.make_img(img_col)
        # result_image = utils.make_img(np.stack([ref, img_col]))
        result_image = ref*0.5 + img_col*0.5
        utils.display_image(result_image/255.0)
        if write_imgs:
            img_col_save = np.clip(np.rint(img_col), 0, 255).astype(np.uint8)
            imageio.imwrite(f'{DIR}/frame{i}{"_wireframe" if wireframe else ""}{"_pose" if reproduce_pose else ""}.png', img_col_save, format='png')
        writer.append_data(np.clip(np.rint(result_image), 0, 255).astype(np.uint8))

        v_f[i] = 0.0

    writer.close()
