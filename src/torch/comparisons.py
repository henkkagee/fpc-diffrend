# imports
from pathlib import Path
import os

import numpy as np
from PIL import Image
import imageio

RESOLUTION = [1600, 1200]

# =================================================================================================
"""
Inside Maya for numerical comparison
"""

# =================================================================================================
"""
Image heatmap comparisons 
"""

def compareSequence(inferred_dir, reference_dir, save_dir):

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(f'{save_dir}/comparison_col.mp4',
                                mode='I', fps=30, codec='libx264', bitrate='16M')
    comp = np.zeros((1600, 1200, 3), dtype=np.uint8)

    for i in range(120):
        imgName = f"frame{i}_pose.png"
        refName = f"pod2colour_pod2texture_{'{:03d}'.format(i)}.tif"
        print(f"img: {imgName} - {refName}")

        img = np.array(Image.open(os.path.join(inferred_dir, imgName)))
        ref = np.array(Image.open(os.path.join(reference_dir, refName)))

        for y in range(1600):
            for x in range(1200):
                imgPix = int(img[y][x])
                refPix = int(ref[y][x])
                diff = imgPix - refPix
                if diff >= 0:
                    comp[y][x][:] = [255, 255 - diff * 2, 255 - diff * 2]
                else:
                    comp[y][x][:] = [255 + diff * 2, 255 + diff * 2, 255]
        comp_save = np.clip(np.rint(comp), 0, 255).astype(np.uint8)
        imageio.imwrite(f'{save_dir}/colcomp_{i}.png', comp_save, format='png')
        writer.append_data(comp_save)

MYDIR = r"W:\thesis\results\safe\d120_prior_1200000_hmc_yoffset\result"
REFDIR = r"W:\thesis\results\reference\d120\pod2colour_pod2texture"
SAVEDIR = r"W:\thesis\results\safe\d120_prior_1200000_hmc_yoffset\result\comp"

compareSequence(MYDIR, REFDIR, SAVEDIR)
