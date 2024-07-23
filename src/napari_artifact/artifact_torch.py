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
import numpy as np
from tqdm import tqdm

@magic_factory(low_pass={"label": "Low Pass Filter", "widget_type": "CheckBox"}, cutoff={"label": "Cutoff Frequency", "widget_type": "FloatSlider", "min": 0.01, "max": 0.5, "step": 0.01})
def fourier_filter_gui_torch(img: Image, low_pass: bool = True, cutoff: float = 0.1, chunk_size: int = 16):
    fourier_filter_worker_torch(img.data, low_pass, cutoff, chunk_size)

def fourier_filter_torch(data: ImageData, low_pass: bool = True, cutoff: float = 0.1, chunk_size: int = 16) -> ImageData:
    print(f"torch used memory(start): {torch.cuda.memory_allocated()}\n")
    if data.ndim == 2:
        torch_data = torch.tensor(data, dtype=torch.float32).to(torch.device("cuda"))
        out_torch_data = _fourier_filter_torch_2d(torch_data,low_pass,cutoff)
        out_data = out_torch_data.cpu().numpy()
    elif data.ndim == 3:
        num_chunks = data.shape[0] // chunk_size + int(data.shape[0] % chunk_size != 0)
        filtered_data = []
        for i in tqdm(range(num_chunks),desc="Processing Chunks"):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, data.shape[0])
            chunk = torch.tensor(data[start_idx:end_idx].copy()).to(torch.device("cuda"))
            filtered_chunk = torch.stack([_fourier_filter_torch_2d(chunk[j], low_pass, cutoff) for j in range(chunk.shape[0])], axis=0)
            filtered_data.append(filtered_chunk.cpu().numpy())
        if chunk_size > 1:
            out_data = np.concatenate(filtered_data,axis=0)
        elif chunk_size == 1:
            out_data = np.stack(filtered_data,axis=0)
        else:
            raise ValueError("Cannot have chunksize < 1")

    elif data.ndim > 3:
        filtered_data = []
        shape_ND = data.shape
        data = data.reshape(-1,data.shape[-2],data.shape[-1]) # This assumes N-dimensional data where the final two dimensions are reserved for ...,H,W
        for i in tqdm(range(data.shape[0]),desc="Processing Chunks"):
            start_idx = i*chunk_size
            end_idx = min((i+1)*chunk_size,data.shape[0])
            chunk = torch.tensor(data[start_idx:end_idx].copy()).to(torch.device("cuda"))
            filtered_chunk = torch.stack([_fourier_filter_torch_2d(chunk[j], low_pass, cutoff) for j in range(chunk.shape[0])], axis=0)
            filtered_data.append(filtered_chunk.cpu().numpy())
        if chunk_size > 1:
            out_data = np.concatenate(filtered_data,axis=0)
        elif chunk_size == 1:
            out_data = np.stack(filtered_data,axis=0)
        else:
            raise ValueError("Cannot have chunksize < 1")
        out_data = out_data.reshape(shape_ND)
    print(f"torch used memory(end): {torch.cuda.memory_allocated()}\n")
    return out_data
    
def _fourier_filter_torch_2d(data: torch.Tensor, low_pass: bool = True, cutoff: float = 0.1) -> torch.Tensor:
    f_transform = fftshift(fft2(data))
    rows, cols = data.shape
    crow, ccol = rows // 2, cols // 2
    mask = torch.zeros((rows, cols), dtype=torch.uint8)
    cutoff = int(cutoff * crow)
    mask = cv2.circle(mask.numpy(), (crow, ccol), cutoff, 1, -1)
    mask = torch.tensor(mask).to(torch.device("cuda"))
    
    if low_pass:
        f_transform = f_transform * mask
    else:
        f_transform = f_transform * (1 - mask)
    
    f_transform = fftshift(f_transform)
    filtered_data = ifft2(f_transform).real
    return filtered_data

@thread_worker(connect={"returned": lambda x: viewer.add_image(x, name="Filtered Image")})
def fourier_filter_worker_torch(data: ImageData, low_pass: bool = True, cutoff: float = 0.1, chunk_size: int = 16) -> ImageData:
    show_info("Filtering image with Fourier filter using PyTorch")
    start = time.time()
    output = fourier_filter_torch(data, low_pass, cutoff, chunk_size)
    processing_time = time.time() - start  # End timing
    show_info(f"Image processing time: {processing_time:.4f} seconds")
    print(f"torch used memory(start): {torch.cuda.memory_allocated  ()}\n")
    torch.cuda.empty_cache()
    print(f"torch used memory(end): {torch.cuda.memory_allocated()}\n")
    # memory_stats()
    return output
