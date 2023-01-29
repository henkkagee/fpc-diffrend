# local
import fit

# ------------------------------------------------------------


def run():
    fit.fitTake(
        max_iter=100000,
        lr_base=10e-4,
        lr_tex_coef=0.5,
        lr_ramp=0.005,
        lr_t=10e-5,
        lr_q=10e-5,
        basemeshpath=r"C:\Users\Henrik\fpc-diffrend\data\basemesh.obj",
        localblpath=r"C:\Users\Henrik\fpc-diffrend\data\ilkvil_blendshapes",
        globalblpath="",
        display_interval=50,
        log_interval=50,
        imdir=r"C:\Users\Henrik\fpc-diffrend\data\reference\dialogue\scene1\take03\20201022_iv_s1_t3_p2col_r1",
        calibpath=r"C:\Users\Henrik\fpc-diffrend\calibration\combined\calibration.json",
        enable_mip=False,
        max_mip_level=6,
        texshape=(1024, 1024, 1),
        out_dir=r"C:\Users\Henrik\fpc-diffrend\data\out\dialogue_120_lapmelmnc_savefix",
        resolution=(1600, 1200),
        mp4_interval=0,
        texpath=r"C:\Users\Henrik\fpc-diffrend\data\ilkka_villi_anchor_greyscale_fix.png",
        maskpath=r"C:\Users\Henrik\fpc-diffrend\data\vertexmasks",
        weight_laplacian=5000,
        weight_meshedge=70,
        meshedge_target=0.05,
        weight_normalconsistency=400
    )

# -----------------------------------------------------------

if __name__ == "__main__":
    run()
