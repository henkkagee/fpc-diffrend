# local
import fit

# ------------------------------------------------------------


def run():
    fit.fitTake(
        max_iter=600,
        lr_base=10e-4,
        lr_ramp=0.1,
        pose_lr=0.2,
        basemeshpath=r"C:\Users\Henkka\Projects\invrend-fpc\data\basemesh.obj",
        localblpath=r"C:\Users\Henkka\Projects\invrend-fpc\data\ilkvil_blendshapes",
        globalblpath="",
        display_interval=10,
        log_interval=25,
        imdir=r"C:\Users\Henkka\Projects\invrend-fpc\data\reference\rom\single10",
        calibpath=r"C:\Users\Henkka\Projects\invrend-fpc\data\calibration\combined\calibration.json",
        enable_mip=False,
        max_mip_level=6,
        texshape=(1024, 1024, 1),
        out_dir=r"C:\Users\Henkka\Projects\invrend-fpc\data\out_img\singlerom",
        resolution=(1600, 1200),
        mp4_interval=5,
        texpath=r"C:\Users\Henkka\Projects\invrend-fpc\data\ilkka_villi_anchor_greyscale_fix.png"
    )

# -----------------------------------------------------------

if __name__ == "__main__":
    run()
