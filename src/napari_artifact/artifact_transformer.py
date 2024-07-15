import torch
import torch.nn as nn
from torch.fft import fft2, ifft2, fftshift
from napari.layers import Image
from napari.utils.notifications import show_info
from napari.types import ImageData
from napari.qt.threading import thread_worker
from napari_cool_tools_io import viewer, memory_stats
from magicgui import magic_factory
import cv2
import time

class FourierFilter(nn.Module):
    def __init__(self, low_pass=True, cutoff=0.1):
        super(FourierFilter, self).__init__()
        self.low_pass = low_pass
        self.cutoff = cutoff

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        f_transform = fftshift(fft2(data))
        rows, cols = data.shape[-2:]
        crow, ccol = rows // 2, cols // 2
        mask = torch.zeros((rows, cols), dtype=torch.uint8)
        cutoff = int(self.cutoff * crow)
        mask = cv2.circle(mask.numpy(), (crow, ccol), cutoff, 1, -1)
        mask = torch.tensor(mask)
        
        if self.low_pass:
            f_transform = f_transform * mask
        else:
            f_transform = f_transform * (1 - mask)
        
        f_transform = fftshift(f_transform)
        filtered_data = ifft2(f_transform).real
        return filtered_data

@magic_factory(low_pass={"label": "Low Pass Filter", "widget_type": "CheckBox"}, cutoff={"label": "Cutoff Frequency", "widget_type": "FloatSlider", "min": 0.01, "max": 0.5, "step": 0.01})
def fourier_filter_gui_torch(img: Image, low_pass: bool = True, cutoff: float = 0.1):
    show_info("Filtering image with Fourier filter using pytorch transformer")
    start = time.time()
    model = FourierFilter(low_pass, cutoff)
    data = torch.tensor(img.data, dtype=torch.float32)
    output = model(data).numpy()
    processing_time = time.time() - start  # End timing
    show_info(f"Image processing time: {processing_time:.4f} seconds")
    viewer.add_image(output, name="Filtered Image")
