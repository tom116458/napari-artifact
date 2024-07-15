import torch
from torch.fft import fft2, ifft2, fftshift
from napari.layers import Image
from napari.utils.notifications import show_info
from napari.types import ImageData
from napari.qt.threading import thread_worker
from napari_cool_tools_io import viewer, memory_stats
from magicgui import magic_factory
import cv2
import time

@magic_factory(low_pass={"label": "Low Pass Filter", "widget_type": "CheckBox"}, cutoff={"label": "Cutoff Frequency", "widget_type": "FloatSlider", "min": 0.01, "max": 0.5, "step": 0.01})
def fourier_filter_gui_torch(img: Image, low_pass: bool = True, cutoff: float = 0.1):
    fourier_filter_worker_torch(img.data, low_pass, cutoff)

def fourier_filter_torch(data: ImageData, low_pass: bool = True, cutoff: float = 0.1) -> ImageData:
    data = torch.tensor(data, dtype=torch.float32)
    f_transform = fftshift(fft2(data))
    rows, cols = data.shape
    crow, ccol = rows // 2, cols // 2
    mask = torch.zeros((rows, cols), dtype=torch.uint8)
    cutoff = int(cutoff * crow)
    mask = cv2.circle(mask.numpy(), (crow, ccol), cutoff, 1, -1)
    mask = torch.tensor(mask)
    
    if low_pass:
        f_transform = f_transform * mask
    else:
        f_transform = f_transform * (1 - mask)
    
    f_transform = fftshift(f_transform)
    filtered_data = ifft2(f_transform).real
    return filtered_data.numpy()

@thread_worker(connect={"yielded": lambda x: viewer.add_image(x, name="Filtered Image")})
def fourier_filter_worker_torch(data: ImageData, low_pass: bool = True, cutoff: float = 0.1) -> ImageData:
    show_info("Filtering image with Fourier filter using PyTorch")
    start = time.time()
    output = fourier_filter_torch(data, low_pass, cutoff)
    processing_time = time.time() - start  # End timing
    show_info(f"Image processing time: {processing_time:.4f} seconds")
    torch.cuda.empty_cache()
    memory_stats()
    yield output
