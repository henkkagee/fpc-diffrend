import xml.etree.ElementTree as et
import numpy as np
tree = et.parse(r"/data/cube/20220310/2021-12-07-01.dicx")
cameras = tree.getroot()[3]
dacam = [x for x in cameras if x.get('name') == 'pod2colour_0001']

intr = dacam[0][0].getchildren()[0].items()
ivals = np.array([float(x[1]) for x in intr]).reshape((3,3))
print(f"ivals: {ivals}")

distCoeffs = dacam[0][0].getchildren()[1].items()
tags = [x[0] for x in distCoeffs]
vals = [float(x[1]) for x in distCoeffs] + [0.0]

print(tags)
print(vals)
print(distCoeffs)