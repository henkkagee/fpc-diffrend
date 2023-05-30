# local
import fit

# ------------------------------------------------------------

"""
Main function for tweaking optimization parameters and paths

"""

def run():
    fit.fitTake(
        max_iter=80000,
        lr_base=10e-4,
        lr_tex_coef=0.5,
        lr_ramp=0.005,
        lr_t=10e-6,
        lr_q=10e-6,
        basemeshpath=r"C:\Users\Henrik\fpc-diffrend\data\basemesh_patched_fixed_verts_and_uv_shifted.obj",
        localblpath=r"C:\Users\Henrik\fpc-diffrend\data\ilkvil_blendshapes_eyes_plusnormals_shifted",
        globalblpath="",
        display_interval=50,
        log_interval=50,
        imdir=r"C:\Users\Henrik\fpc-diffrend\data\reference\rom\20191106_ilkvil_ML_ROM_take0001_pod2colour_range03",
        calibpath=r"C:\Users\Henrik\fpc-diffrend\calibration\calibration.json",
        enable_mip=False,
        max_mip_level=6,
        texshape=(1024, 1024, 1),
        out_dir=r"C:\Users\Henrik\fpc-diffrend\data\out\ROM2_100000_9cams_prior_yoffset",
        resolution=(1600, 1200),
        mp4_interval=200,
        texpath=r"C:\Users\Henrik\fpc-diffrend\data\texture_g_edit.png",
        tex_startlearnratio=20,
        free_startlearnratio=4,
        tex_ramplearnratio=[2, 3/4],
        maskpath=r"C:\Users\Henrik\fpc-diffrend\data\vertexmasks",
        weight_laplacian=5000,
        weight_meshedge=0,  # 70
        meshedge_target=0.05,
        weight_normalconsistency=0,  # 400
        cam_idxs=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        whiten_mean=10,
        whiten_std=2,
        mode="prior",
        combined_corrective_coefficient=0.5,
        regularize_correctives=False,
        regularize_prior=False
    )

# -----------------------------------------------------------

if __name__ == "__main__":
    run()
