# builtin
import os
import json

# 3rd party
import numpy as np
from PIL import Image
import imageio

# local
import src.torch.data as data
import src.torch.utils as utils
import src.torch.camera as camera

# -------------------------------------------------------------------------------------------------


# Get objs
DIR = r"C:\Users\Henrik\fpc-diffrend\data\reference\dialogue\scene1\take03\20201022_iv_s1_t3_p2col_r1\pod2colour_pod2texture"
imgs = os.listdir(DIR)

writer = imageio.get_writer(f'{DIR}/result_wireframe.mp4', mode='I', fps=30, codec='libx264', bitrate='16M')

for i in range(0, 120):
    img = np.array(Image.open(os.path.join(DIR, f"pod2colour_pod2texture_{i:03d}.tif")))
    # utils.display_image(img)
    # imageio.imwrite(f'{DIR}/frame{i}.png', img_col, format='png')
    writer.append_data(np.clip(np.rint(img), 0, 255).astype(np.uint8))

writer.close()
