# local
import fit

# ------------------------------------------------------------


def run():
    fit.fitTake(
        max_iter=50,
        lr_base=10e-3,
        lr_ramp=0.1,
        basemeshpath=r"C:\Users\Henkka\Projects\invrend-fpc\data\basemesh.obj",
        localblpath=r"C:\Users\Henkka\Projects\invrend-fpc\data\ilkvil_blendshapes",
        globalblpath="",
        display_interval=1,
        log_interval=50,
        imdir=r"C:\Users\Henkka\Projects\invrend-fpc\data\reference\rom\take0001\20191106_ilkvil_ML_ROM_take0001_pod2colour_range01",
        calibpath=r"C:\Users\Henkka\Projects\invrend-fpc\data\calibration\combined\calibration.json",
        enable_mip=False,
        max_mip_level=6,
        texshape=(1024, 1024, 1),
        out_dir=r"C:\Users\Henkka\Projects\invrend-fpc\data\out_img\take0001",
        resolution=(1600, 1200),
        mp4_interval=3,
        texpath=r"C:\Users\Henkka\Projects\invrend-fpc\data\ilkka_villi_anchor_greyscale_fix.png"
    )

# -----------------------------------------------------------

if __name__ == "__main__":
    run()
