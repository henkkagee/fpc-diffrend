# local
import fit

# ------------------------------------------------------------


def run():
    fit.fitTake(
        max_iter=100,
        lr_base=10e-4,
        lr_ramp=0.1,
        pose_lr=0.2,
        basemeshpath=r"C:\Users\Henkka\Projects\invrend-fpc\data\basemesh.obj",
        localblpath=r"C:\Users\Henkka\Projects\invrend-fpc\data\ilkvil_blendshapes",
        globalblpath="",
        display_interval=10,
        log_interval=10,
        imdir=r"C:\Users\Henkka\Projects\invrend-fpc\data\reference\rom\take0001\20191106_ilkvil_ML_ROM_take0001_pod2colour_range01",
        calibpath=r"C:\Users\Henkka\Projects\invrend-fpc\data\calibration\combined\calibration.json",
        enable_mip=False,
        max_mip_level=6,
        texshape=(1024, 1024, 1),
        out_dir=r"C:\Users\Henkka\Projects\invrend-fpc\data\out_img\take0001",
        resolution=(1600, 1200),
        mp4_interval=98,
        texpath=r"C:\Users\Henkka\Projects\invrend-fpc\data\ilkka_villi_anchor_greyscale_fix.png"
    )

# -----------------------------------------------------------

if __name__ == "__main__":
    run()
