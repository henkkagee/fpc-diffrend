# Facial Performance Capture through Differentiable Rendering with a Facial Rig Prior

![Comparison of reference video and inferred facial animation rendered from the same perspective](doc/comparison_texture.gif)

## Licensing

This software uses the [Nvdiffrast](https://github.com/NVlabs/nvdiffrast) library according to the [Nvidia Source Code License](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt) under non-commercial terms.

## Short description

This project aims to solve markerless facial performance capture through differentiable rendering using a facial rig prior by Remedy Entertainment. The software is built on Nvdiffrast and PyTorch.

This work is my Master's thesis at Aalto University and made possible in cooperation with Remedy Entertainment. The project is made open-source to share knowledge on facial performance capture in the industry.

Accompanying reference and rig data for reproducing the results will later be published here. The thesis itself is published at [aaltodoc.aalto.fi](aaltodoc.aalto.fi) once it is graded.

## Abstract

Realistic facial animation is an important component in creating believable
characters in digital media. Facial performance capture attempts to solve this need
by recording actor performances and reproducing them digitally. One approach in
performance capture is the use of differentiable rendering: an analysis-by-synthesis
approach where candidate images are rendered and compared against reference
material to optimize scene parameters such as geometry and texture in order to
match the reference as closely as possible. A differentiable renderer makes this possible
by computing the gradient of an objective function over the scene parameters.

This Thesis aims to explore differentiable rendering for inferring facial animation
from markerless multi-view reference video footage. This has been done before, but the
approaches have not been directly applicable in the video game industry where head
stabilization data and a semantically meaningful animation basis is crucial for further
processing. To obtain these advantages we leverage a highly tailored facial rig as prior
data: a facial model based on blendshapes, parametrized as a linear combination of
meshes representing a range of facial expressions, common in the industry to control
character animation. We design and implement a facial performance capture pipeline
for Remedy Entertainment as an open-source contribution and infer animation with
varying configurations. The underlying optimization architecture is built on Nvidia’s
nvdiffrast-library, a Python- and PyTorch-based differentiable rendering framework
that utilizes highly-optimized graphics pipelines on GPUs to accelerate performance.

Experiments with the implemented pipeline show that staying completely onmodel 
with a blendshape-based facial rig as prior data provides both advantages
and disadvantages. Although we propose numerous improvements, the animation
quality is of insufficient cinematic quality, particularly with more extreme facial
expressions. However, we benefit from shape constraints set by our rig prior and
see computation simplicity and performance gains by only learning per-frame shape
activations, instead of the shapes themselves, as done in previous works. Moreover,
we obtain head stabilization data that is important to have down the line in video
game production, and the use of blendshapes as the basis of our resulting animation
enables semantically meaningful editing after inference.

![Inferred animation rendered from nine camera angles using wireframe](doc/wireframe_grid.gif)

## Structure of the repository

This optimizer is written on top of Nvdiffrast and PyTorch. The main function is found in src/torch/main.py, where all 
configurables and paths live. This in turn calls fit.py, which does the heavy lifting and where the main optimization 
loop lives. Camera projections and transformations live in src/torch/camera.py, while data operations are in src/torch/data.py. 
Miscallaneous operations are in src/torch/utils.py The rest of the files there are for rendering result videos and comparisons.

The scripts living in the torch/ directory are for calibrating cameras using OpenCv2.

The src/matlab/ directory contains matlab code for exctracting image sequences from Norpix SEQ files. This code is not 
written by the author here, but by Paul Siefert.

Camera calibrations are found in calibration/calibration.json.

More visualizations are in doc/.
