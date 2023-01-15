# std
import os
import json

# 3rd party
import nvdiffrast.torch as dr
import numpy as np
import torch
import imageio
from PIL import Image
import cv2 as cv2

# local
import src.torch.utils as utils
import src.torch.data as data
import src.torch.camera as camera

# ----------------------------------------------------------------------------
# Quaternion math.
# ----------------------------------------------------------------------------


# Unit quaternion.
def q_unit():
    return np.asarray([1, 0, 0, 0], np.float32)

# ----------------------------------------------------------------------------


# Get a random normalized quaternion.
def q_rnd():
    u, v, w = np.random.uniform(0.0, 1.0, size=[3])
    v *= 2.0 * np.pi
    w *= 2.0 * np.pi
    return np.asarray([(1.0-u)**0.5 * np.sin(v), (1.0-u)**0.5 * np.cos(v), u**0.5 * np.sin(w), u**0.5 * np.cos(w)], np.float32)

# ----------------------------------------------------------------------------


# Get a random quaternion from the octahedral symmetric group S_4.
_r2 = 0.5**0.5
_q_S4 = [[ 1.0, 0.0, 0.0, 0.0], [ 0.0, 1.0, 0.0, 0.0], [ 0.0, 0.0, 1.0, 0.0], [ 0.0, 0.0, 0.0, 1.0],
         [-0.5, 0.5, 0.5, 0.5], [-0.5,-0.5,-0.5, 0.5], [ 0.5,-0.5, 0.5, 0.5], [ 0.5, 0.5,-0.5, 0.5],
         [ 0.5, 0.5, 0.5, 0.5], [-0.5, 0.5,-0.5, 0.5], [ 0.5,-0.5,-0.5, 0.5], [-0.5,-0.5, 0.5, 0.5],
         [ _r2,-_r2, 0.0, 0.0], [ _r2, _r2, 0.0, 0.0], [ 0.0, 0.0, _r2, _r2], [ 0.0, 0.0,-_r2, _r2],
         [ 0.0, _r2, _r2, 0.0], [ _r2, 0.0, 0.0,-_r2], [ _r2, 0.0, 0.0, _r2], [ 0.0,-_r2, _r2, 0.0],
         [ _r2, 0.0, _r2, 0.0], [ 0.0, _r2, 0.0, _r2], [ _r2, 0.0,-_r2, 0.0], [ 0.0,-_r2, 0.0, _r2]]


def q_rnd_S4():
    return np.asarray(_q_S4[np.random.randint(24)], np.float32)

# ----------------------------------------------------------------------------


# Quaternion slerp.
def q_slerp(p, q, t):
    d = np.dot(p, q)
    if d < 0.0:
        q = -q
        d = -d
    if d > 0.999:
        a = p + t * (q-p)
        return a / np.linalg.norm(a)
    t0 = np.arccos(d)
    tt = t0 * t
    st = np.sin(tt)
    st0 = np.sin(t0)
    s1 = st / st0
    s0 = np.cos(tt) - d*s1
    return s0*p + s1*q

# ----------------------------------------------------------------------------


# Quaterion scale (slerp vs. identity quaternion).
def q_scale(q, scl):
    return q_slerp(q_unit(), q, scl)

# ----------------------------------------------------------------------------


# Quaternion product.
def q_mul(p, q):
    s1, V1 = p[0], p[1:]
    s2, V2 = q[0], q[1:]
    s = s1*s2 - np.dot(V1, V2)
    V = s1*V2 + s2*V1 + np.cross(V1, V2)
    return np.asarray([s, V[0], V[1], V[2]], np.float32)

# ----------------------------------------------------------------------------


# Angular difference between two quaternions in degrees.
def q_angle_deg(p, q):
    p = p.detach().cpu().numpy()
    q = q.detach().cpu().numpy()
    d = np.abs(np.dot(p, q))
    d = min(d, 1.0)
    return np.degrees(2.0 * np.arccos(d))

# ----------------------------------------------------------------------------


# Quaternion product
def q_mul_torch(p, q):
    a = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    b = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    c = p[0]*q[2] + p[2]*q[0] + p[3]*q[1] - p[1]*q[3]
    d = p[0]*q[3] + p[3]*q[0] + p[1]*q[2] - p[2]*q[1]
    return torch.stack([a, b, c, d])

# ----------------------------------------------------------------------------


# Convert quaternion to 4x4 rotation matrix.
def q_to_mtx(q):
    r0 = torch.stack([1.0-2.0*q[1]**2 - 2.0*q[2]**2, 2.0*q[0]*q[1] - 2.0*q[2]*q[3], 2.0*q[0]*q[2] + 2.0*q[1]*q[3]])
    r1 = torch.stack([2.0*q[0]*q[1] + 2.0*q[2]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[2]**2, 2.0*q[1]*q[2] - 2.0*q[0]*q[3]])
    r2 = torch.stack([2.0*q[0]*q[2] - 2.0*q[1]*q[3], 2.0*q[1]*q[2] + 2.0*q[0]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[1]**2])
    rr = torch.transpose(torch.stack([r0, r1, r2]), 1, 0)
    rr = torch.cat([rr, torch.tensor([[0], [0], [0]], dtype=torch.float32).cuda(device='cuda')], dim=1) # Pad right column.
    rr = torch.cat([rr, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).cuda(device='cuda')], dim=0)  # Pad bottom row.
    return rr

# ----------------------------------------------------------------------------


def render(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution: tuple):
    # Setup TF graph for reference.
    pos_clip    = camera.transform_clip(mtx, pos)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution[0], resolution[1]])
    texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
    colour = dr.texture(tex[None, ...], texc, filter_mode='linear')
    colour       = dr.antialias(colour, rast_out, pos_clip, pos_idx)
    return colour

# ----------------------------------------------------------------------------
# Cube pose fitter.
# ----------------------------------------------------------------------------


def fit_pose(max_iter           = 10000,
             repeats            = 1,
             log_interval       = 10,
             display_interval   = None,
             display_res        = 512,
             lr_base            = 0.01,
             lr_falloff         = 1.0,
             nr_base            = 1.0,
             nr_falloff         = 1e-4,
             grad_phase_start   = 0.5,
             resolution         = (256, 256),
             out_dir            = None,
             imdir              = "",
             basemeshpath       = "",
             texpath            = "",
             datadir            = "",
             calibs             = {},
             log_fn             = None,
             mp4save_interval   = None,
             mp4save_fn         = None):

    cams = os.listdir(imdir)
    n_frames = utils.assert_num_frames(cams, imdir)

    # Set up logging.
    if out_dir:
        print(f'Saving results under {out_dir}')
        writer = imageio.get_writer(f'{out_dir}/progress.mp4', mode='I', fps=30, codec='libx264', bitrate='16M')
    else:
        out_dir = None
        print('No output directory specified, not saving log or images')
    gl_avg = []
    log_file = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if log_fn:
            log_file = open(f'{out_dir}/{log_fn}', 'wt')

    # object data
    rubiks = data.MeshData(basemeshpath)
    vtx_pos = torch.tensor(rubiks.vertices, dtype=torch.float32, device='cuda')
    pos_idx = torch.tensor(rubiks.faces, dtype=torch.int32, device='cuda')
    uv_idx = torch.tensor(rubiks.fuv, dtype=torch.int32, device='cuda')
    uv = torch.tensor(rubiks.uv, dtype=torch.float32, device='cuda')
    texture = np.array(Image.open(texpath)) / 255.0
    tex = torch.tensor(texture, dtype=torch.float32, device='cuda')

    print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], vtx_pos.shape[0]))
    pose_init = q_rnd()
    pose_opt = torch.tensor(pose_init / np.sum(pose_init ** 2) ** 0.5, dtype=torch.float32, device='cuda',
                            requires_grad=True)
    vtx_pos_split = torch.reshape(vtx_pos, (vtx_pos.shape[0] // 3, 3))

    loss_best = np.inf
    pose_best = pose_opt.detach().clone()

    glctx = dr.RasterizeGLContext()
    optimizer = torch.optim.Adam([pose_opt], betas=(0.9, 0.999), lr=lr_base)

    for cam in cams:
        if cam != "_pod2texture":
            continue
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
            img = cv2.undistort(img, intr, dist)
            # ref = torch.from_numpy(np.flip(img, 0).copy()).cuda(device='cuda')
            ref = torch.from_numpy(img).cuda(device='cuda')

            # lens distortion handled as preprocess in reference images
            projection = torch.tensor(camera.intrinsic_to_projection(intr), dtype=torch.float32, device='cuda')
            # projection = torch.tensor(camera.default_projection(), dtype=torch.float32, device='cuda')
            modelview = torch.tensor(camera.extrinsic_to_modelview(rot, trans), dtype=torch.float32, device='cuda')

            # Render.
            for it in range(max_iter + 1):
                # Set learning rate.
                itf = 1.0 * it / max_iter
                nr = nr_base * nr_falloff**itf
                lr = lr_base * lr_falloff**itf
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # Noise input.
                if itf >= grad_phase_start:
                    noise = q_unit()
                else:
                    noise = q_scale(q_rnd(), nr)
                    noise = q_mul(noise, q_rnd_S4()) # Orientation noise.

                mvp = torch.matmul(projection, modelview)

                # Render.
                pose_total_opt = q_mul_torch(pose_opt, noise)
                mtx_total_opt  = torch.matmul(mvp, q_to_mtx(pose_total_opt))
                color_opt      = render(glctx, mtx_total_opt, vtx_pos_split, pos_idx, uv, uv_idx, tex, resolution)

                # Image-space loss.
                diff = (color_opt - ref)**2 # L2 norm.
                diff = torch.tanh(5.0 * torch.max(diff, dim=-1)[0])
                loss = torch.mean(diff)

                # Measure image-space loss and update best found pose.
                loss_val = float(loss)
                if (loss_val < loss_best) and (loss_val > 0.0):
                    pose_best = pose_total_opt.detach().clone()
                    loss_best = loss_val
                    if itf < grad_phase_start:
                        with torch.no_grad(): pose_opt[:] = pose_best

                # Print/save log.
                if log_interval and (it % log_interval == 0):
                    s = "iter=%d,loss=%f,loss_best=%f,lr=%f,nr=%f" % (it, loss_val, loss_best, lr, nr)
                    print(s)
                    if log_file:
                        log_file.write(s + "\n")

                # Run gradient training step.
                if itf >= grad_phase_start:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    pose_opt /= torch.sum(pose_opt**2)**0.5

                # Show/save image.
                display_image = display_interval and (it % display_interval == 0)
                save_mp4      = mp4save_interval and (it % mp4save_interval == 0)

                if display_image or save_mp4:
                    img_ref  = ref.detach().cpu().numpy()
                    img_ref = np.array(img_ref.copy(), dtype=np.float32)/255
                    img_opt  = color_opt[0].detach().cpu().numpy()
                    img_best = render(glctx, torch.matmul(mvp, q_to_mtx(pose_best)), vtx_pos_split, pos_idx, uv, uv_idx, tex,
                                      resolution)[0].detach().cpu().numpy()
                    result_image = np.concatenate([img_ref, img_best, img_opt], axis=1)
                    # print(f"shapes: img_ref: {img_ref.shape}, img_opt: {img_opt.shape}, img_best: {img_best.shape}")

                    if display_image:
                        utils.display_image(result_image, size=display_res, title='%d / %d' % (it, max_iter))
                    if save_mp4:
                        writer.append_data(np.clip(np.rint(result_image*255.0), 0, 255).astype(np.uint8))

    # Done.
    if writer is not None:
        writer.close()
    if log_file:
        log_file.close()

#----------------------------------------------------------------------------

def main():

    path = r"C:\Users\Henrik\fpc-diffrend\calibration\2018-11-15\calibration_test_DI.json"
    with open(path) as json_file:
        calibs = json.load(json_file)

    # Run.
    fit_pose(
        max_iter=2000,
        repeats=1,
        log_interval=10,
        display_interval=5,
        out_dir=r"C:/Users/Henrik/fpc-diffrend/data/out/cube2",
        log_fn='log.txt',
        mp4save_interval=10,
        mp4save_fn='progress.mp4',
        basemeshpath=r"C:\Users\Henrik\fpc-diffrend\data\cube\rubiks_fixed.obj",
        imdir=r"C:\Users\Henrik\fpc-diffrend\data\cube\20220310\neutrals\rubik_cube\neutral\take0001\fullres",
        calibs=calibs,
        texpath=r"C:\Users\Henrik\fpc-diffrend\data\cube\rubiks.png",
        resolution=(1600, 1200)
    )

    # Done.
    print("Done.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
