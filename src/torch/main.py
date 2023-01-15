# local
import fit

# ------------------------------------------------------------


def run():
    fit.fitTake(
        max_iter=80000,
        lr_base=10e-4,
        lr_ramp=0.1,
        pose_lr=0.2,
        cam_iter=1,
        basemeshpath=r"C:\Users\Henrik\fpc-diffrend\data\basemesh.obj",
        localblpath=r"C:\Users\Henrik\fpc-diffrend\data\ilkvil_blendshapes",
        globalblpath="",
        display_interval=50,
        log_interval=50,
        imdir=r"C:\Users\Henrik\fpc-diffrend\data\reference\dialogue\scene1\take03\20201022_iv_s1_t3_p2col_r3_short",
        calibpath=r"C:\Users\Henrik\fpc-diffrend\calibration\combined\calibration.json",
        enable_mip=False,
        max_mip_level=6,
        texshape=(1024, 1024, 1),
        out_dir=r"C:\Users\Henrik\fpc-diffrend\data\out\dialogue_sc1_t3_short",
        resolution=(1600, 1200),
        mp4_interval=50,
        texpath=r"C:\Users\Henrik\fpc-diffrend\data\ilkka_villi_anchor_greyscale_fix.png",
        maskpath=r"C:\Users\Henrik\fpc-diffrend\data\vertexmasks",
    )

# -----------------------------------------------------------

if __name__ == "__main__":
    run()
