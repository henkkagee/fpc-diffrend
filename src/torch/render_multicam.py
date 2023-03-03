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

optimname = "d120_prior_1200000_hmc_yoffset"
wireframe = True
n_frames = 120
reproduce_pose = True

# Get objs
REFDIR = r"C:\Users\Henrik\fpc-diffrend\data\reference\dialogue\scene1\take03\20201022_iv_s1_t3_p2col_r1\pod2colour_pod2texture"
DIR = r"C:\Users\Henrik\fpc-diffrend\data\out\{}\result".format(optimname)
objs = os.listdir(DIR)
if wireframe:
    texpath = os.path.join(DIR, "ilkka_villi_anchor_greyscale_fix_wireframe.png")
else:
    texpath = os.path.join(DIR, "texture.png")

camNames = ["pod1secondary", "pod2secondary", "pod3secondary",
            "pod1texture", "pod2texture", "pod3texture",
            "pod1primary", "pod2primary", "pod3primary"]

# common mesh info
basemesh = data.MeshData(os.path.join(DIR, "basemesh.obj"))
pos_idx = torch.tensor(basemesh.faces, dtype=torch.int32, device='cuda')
uv = torch.tensor(basemesh.uv, dtype=torch.float32, device='cuda')
uv_idx = torch.tensor(basemesh.fuv, dtype=torch.int32, device='cuda')
tex = np.array(Image.open(texpath))/255.0
tex = tex[..., np.newaxis]
tex = np.flip(tex, 0)
tex = torch.tensor(tex.copy(), dtype=torch.float32, device='cuda', requires_grad=True)
resolution = (800, 600)

glctx = dr.RasterizeGLContext(device='cuda')

# get camera calibration
calibpath = r"C:\Users\Henrik\fpc-diffrend\calibration\combined\calibration.json"
cam = "pod2texture"
with open(calibpath) as json_file:
    calibs = json.load(json_file)
calib = calibs[cam]
intr = np.asarray(calib['intrinsic'], dtype=np.float32)
dist = np.asarray(calib['distortion'], dtype=np.float32)
rot = np.asarray(calib['rotation'], dtype=np.float32)
trans_calib = np.asarray(calib['translation'], dtype=np.float32)

# saved pose tensors
obj_text = codecs.open(os.path.join(DIR, 'pose.json'), 'r', encoding='utf-8').read()
dictobj = json.loads(obj_text)
pose_trans = torch.tensor(dictobj['translation'], device='cuda')
pose_rot = torch.tensor(dictobj['rotation'], device='cuda')

glctx = dr.RasterizeGLContext(device='cuda')
writer = imageio.get_writer(f'{DIR}/result_multicams{"_wireframe" if wireframe else ""}{"_pose" if reproduce_pose else ""}.mp4', mode='I', fps=30, codec='libx264', bitrate='16M')

"""for i, obj in enumerate(objs):
    if "basemesh" in obj:
        continue
    # get vertices
    vertices = []
    with open(os.path.join(DIR, obj), 'r') as f:"""

v_f = torch.zeros(n_frames, dtype=torch.float32, device='cuda')

for i in range(0, 120):

    v_f[i] = 1.0

    vertices = []
    # ref = np.array(Image.open(os.path.join(REFDIR, f"pod2colour_pod2texture_{i:03d}.tif")))
    # ref = ref.reshape((ref.shape[0], ref.shape[1], 1))
    with open(os.path.join(DIR, f"{i}.obj"), 'r') as f:
        for line in f:
            if line.startswith("v "):
                vertices.extend([float(x) for x in line.strip().split(" ")[1:]])
            elif line.startswith("vt "):
                break

    vtx_pos = torch.tensor(vertices, dtype=torch.float32, device='cuda')

    imgs = []

    for cam in camNames:
        calib = calibs[cam]
        intr = np.asarray(calib['intrinsic'], dtype=np.float32)
        dist = np.asarray(calib['distortion'], dtype=np.float32)
        rot = np.asarray(calib['rotation'], dtype=np.float32)
        trans_calib = np.asarray(calib['translation'], dtype=np.float32)

        projection = camera.intrinsic_to_projection(intr)
        proj = torch.from_numpy(projection).cuda(device='cuda')
        modelview = camera.extrinsic_to_modelview(rot, trans_calib)
        trans = torch.tensor(camera.translate(0.0, 170.0, 0.0), dtype=torch.float32, device='cuda')
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
        imgs.append(img_col)

    # result_image = utils.make_img(img_col)
    # result_image = utils.make_img(np.stack([ref, img_col]))
    result_image = utils.make_img(np.stack(imgs), ncols=3)
    utils.display_image(result_image/255.0)
    # imageio.imwrite(f'{DIR}/frame{i}.png', img_col, format='png')
    writer.append_data(np.clip(np.rint(result_image), 0, 255).astype(np.uint8))

    v_f[i] = 0.0

writer.close()
