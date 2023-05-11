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

def compareSequence(inferred_dir, reference_dir, save_dir, colour=True):

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(f'{save_dir}/comparison_col.mp4',
                                mode='I', fps=30, codec='libx264', bitrate='16M')
    comp = np.zeros((1600, 1200, 3), dtype=np.uint8)

    for i in range(120):
        imgName = f"frame{i}_pose.png"
        refName = f"pod2colour_pod2primary_{'{:03d}'.format(i)}.tif"
        print(f"img: {imgName} - {refName}")

        img = np.array(Image.open(os.path.join(inferred_dir, imgName)))
        ref = np.array(Image.open(os.path.join(reference_dir, refName)))

        for y in range(1600):
            for x in range(1200):
                imgPix = int(img[y][x])
                refPix = int(ref[y][x])
                if colour:
                    diff = imgPix - refPix
                    if diff >= 0:
                        comp[y][x][:] = [255, 255 - diff * 2, 255 - diff * 2]
                    else:
                        comp[y][x][:] = [255 + diff * 2, 255 + diff * 2, 255]
                else:
                    diff = abs(imgPix - refPix)
                    comp[y][x][:] = [255 - diff * 2, 255 - diff * 2, 255 - diff * 2]
        comp_save = np.clip(np.rint(comp), 0, 255).astype(np.uint8)
        imageio.imwrite(f'{save_dir}/colcomp_{i}.png', comp_save, format='png')
        writer.append_data(comp_save)


def compareSequenceNumerical(inferred_dir, reference_dir, save_dir, colour=True):

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(save_dir, "numerical_clip.csv"), "w") as savefile:
        fileDiffs = []
        for i in range(120):
            imgName = f"frame{i}_pose.png"
            refName = f"pod2colour_pod2primary_{'{:03d}'.format(i)}.tif"
            print(f"img: {imgName} - {refName}")

            img = np.array(Image.open(os.path.join(inferred_dir, imgName))).astype(np.int32)
            ref = np.array(Image.open(os.path.join(reference_dir, refName))).astype(np.int32)

            pixelDiffs = []
            rowMeans = []
            for y in range(1600):
                if y < 200:
                    continue
                elif y > 1400:
                    break
                rowDiff = abs(img[y][100:1100] - ref[y][100:1100])
                rowMeans.append(np.mean(rowDiff))
            imgMean = np.mean(np.array(rowMeans))
            print(f"img {i}: {imgMean}")
            fileDiffs.append(imgMean)
            savefile.write(f"{str(imgMean)}, {', '.join([str(m) for m in rowMeans])}\n")
        savefile.write(str(np.mean(np.array(fileDiffs))))
        print(f"total: {fileDiffs}")

MYDIR = r"W:\thesis\results\safe_final\d120_combined_150000_heavier_laplacian_2\result"
REFDIR = r"W:\thesis\results\reference\d120\pod2colour_pod2primary"
SAVEDIR = r"W:\thesis\results\safe_final\d120_combined_150000_heavier_laplacian_2\result\compbw"

compareSequenceNumerical(MYDIR, REFDIR, SAVEDIR, colour=False)
