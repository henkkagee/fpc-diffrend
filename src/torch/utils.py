import os
import statistics

from PIL import Image
import numpy as np


# -------------------------------------------------------------------------------------------------

def gaussian_filter(arr, kernel_radius):
    """
    Gaussian filter for smoothing out the optimization landscape. Might not work like this
    since gradients are probably not preserved through the pixel assignment...

    :param arr: 2D tensor to filter
    :param kernel_radius: Kernel size in each direction from the center
    :return:
    """
    for j in range(arr.size[0]):
        top = max(j - kernel_radius, 0)
        bottom = min(j + kernel_radius, arr.size[0] - 1)
        for i in range(arr.size[1]):
            left = max(i - kernel_radius, 0)
            right = min(i + kernel_radius, arr.size[1] - 1)
            vals = []
            for v in range(top, bottom+1):
                for u in range(left, right+1):
                    vals.append(arr[v][u])
            mean = statistics.mean(vals)
            arr[j][i] = mean

# -------------------------------------------------------------------------------------------------


def assert_num_frames(cams, imdir):
    """
    Get number of frames outside optimization loop to init tensors and make sure each camera has the
    same number of frames

    :param cams: List of camera directories with stored reference frames
    :return:
    """
    n_frames = []
    for cam in cams:
        camdir = os.path.join(imdir, cam)
        frames = os.listdir(camdir)
        n_frames.append(len(frames))
    assert not any([x != n_frames[0] for x in n_frames])
    return n_frames[0]

# -------------------------------------------------------------------------------------------------


def make_img(arr, ncols=2):
    """
    Stack a number of images into a grid.

    :param arr: Array of images of same shape
    :param ncols: Number of columns in image grid.
    :return:
    """
    n, height, width, nc = arr.shape
    nrows = n//ncols
    assert n == nrows*ncols
    return arr.reshape(nrows, ncols, height, width, nc).swapaxes(1,2).reshape(height*nrows, width*ncols, nc)

# -------------------------------------------------------------------------------------------------


_glfw_window = None
def display_image(image, zoom=None, size=None, title=None): # HWC
    # Import OpenGL and glfw.
    import OpenGL.GL as gl
    import glfw

    # Zoom image if requested.
    image = np.asarray(image)
    if size is not None:
        assert zoom is None
        zoom = max(1, size // image.shape[0])
    if zoom is not None:
        image = image.repeat(zoom, axis=0).repeat(zoom, axis=1)
    height, width, channels = image.shape

    # Initialize window.
    if title is None:
        title = 'Debug window'
    global _glfw_window
    if _glfw_window is None:
        glfw.init()
        _glfw_window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(_glfw_window)
        glfw.show_window(_glfw_window)
        glfw.swap_interval(0)
    else:
        glfw.make_context_current(_glfw_window)
        glfw.set_window_title(_glfw_window, title)
        glfw.set_window_size(_glfw_window, width, height)

    # Update window.
    glfw.poll_events()
    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glWindowPos2f(0, 0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl_format = {3: gl.GL_RGB, 2: gl.GL_RG, 1: gl.GL_LUMINANCE}[channels]
    gl_dtype = {'uint8': gl.GL_UNSIGNED_BYTE, 'float32': gl.GL_FLOAT}[image.dtype.name]
    gl.glDrawPixels(width, height, gl_format, gl_dtype, image[::-1])
    glfw.swap_buffers(_glfw_window)
    if glfw.window_should_close(_glfw_window):
        return False
    return True

# -------------------------------------------------------------------------------------------------


def save_image(filepath, x):
    img = Image.fromarray(x)
    img.save(filepath)