import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift
from napari.layers import Image
import napari
from napari.utils.notifications import show_info
from napari.layers import Image, Layer
from napari.types import ImageData
from napari.qt.threading import thread_worker
from napari_cool_tools_io import torch, viewer, device, memory_stats
from magicgui import magic_factory
import cv2


@magic_factory(low_pass={"label": "Low Pass Filter", "widget_type": "CheckBox"}, cutoff={"label": "Cutoff Frequency", "widget_type": "FloatSlider", "min": 0.01, "max": 0.5, "step": 0.01})
def fourier_filter_gui(img: Image, low_pass: bool = True, cutoff: float = 0.1):
    fourier_filter_worker(img.data, low_pass, cutoff)

def fourier_filter(data: ImageData, low_pass: bool = True, cutoff: float = 0.1) -> ImageData:
    """Implementation of Fourier filter function
    Args:
        data (ImageData): Image/Volume to be filtered.
        low_pass (bool): Flag indicating whether to use low pass filter. If False, high pass filter is used.
        cutoff (float): cutoff frequency for filter
    Returns:
        ImageData that has had Fourier filter applied to it.
    """
    # Calculate the 2D Fourier transform of the input image
    f_transform = fft2(data)
    # Shift the zero frequency component to the center
    f_transform = fftshift(f_transform)
    # Get the shape of the image data
    rows, cols = data.shape
    # Calculate the center of the image
    crow, ccol = rows // 2, cols // 2
    # Create a mask with the same shape as the image data
    mask = np.zeros((rows, cols), np.uint8)
    # Calculate the cutoff frequency
    cutoff = int(cutoff * crow)
    # Create a circular mask
    cv2.circle(mask, (crow, ccol), cutoff, 1, -1)
    # Apply the mask to the Fourier transform
    if low_pass:
        f_transform = f_transform * mask
    else:
        f_transform = f_transform * (1 - mask)
    # Shift the zero frequency component back to the corner
    f_transform = fftshift(f_transform)
    # Calculate the inverse Fourier transform
    filtered_data = ifft2(f_transform)
    # Return the real part of the filtered data
    return np.real(filtered_data)

@thread_worker(connect={"yielded": lambda x: viewer.add_image(x, name="Filtered Image")})
def fourier_filter_worker(data: ImageData, low_pass: bool = True, cutoff: float = 0.1) -> ImageData:
    show_info("Filtering image with Fourier filter")
    output = fourier_filter(data, low_pass, cutoff)
    torch.cuda.empty_cache()
    memory_stats()
    yield output