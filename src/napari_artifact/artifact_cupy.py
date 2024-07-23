import cupy as cp
from typing import Generator
from cupyx.scipy.fft import fft2, ifft2, fftshift
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
def fourier_filter_gui_cupy(img: Image, low_pass: bool = True, cutoff: float = 0.1, chunk_size: int = 16):
    fourier_filter_worker_cupy(img.data, low_pass, cutoff, chunk_size)


def fourier_filter_cupy(data: ImageData, low_pass: bool = True, cutoff: float = 0.1,chunk_size: int = 16) -> ImageData:
    print(f"cupy used memory(start): {cp.get_default_memory_pool().used_bytes()} bytes\n")
    if data.ndim == 2:
        cp_data = cp.asarray(data)
        out_cp = _fourier_filter_cupy_2d(cp_data,low_pass,cutoff)
        out_data = cp.asnumpy(out_cp)
    elif data.ndim == 3:
        num_chunks = data.shape[0] // chunk_size + int(data.shape[0] % chunk_size != 0)
        filtered_data = []
        for i in tqdm(range(num_chunks),desc="Processing Chunks"):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, data.shape[0])
            chunk = cp.asarray(data[start_idx:end_idx])
            filtered_chunk = cp.stack([_fourier_filter_cupy_2d(chunk[j], low_pass, cutoff) for j in range(chunk.shape[0])], axis=0)
            filtered_data.append(cp.asnumpy(filtered_chunk))
        if chunk_size > 1:
            out_data = np.concatenate(filtered_data,axis=0)
        elif chunk_size == 1:
            out_data = np.stack(filtered_data,axis=0)
        else:
            raise ValueError("Cannot have chunksize < 1")
        
        """
        for img in data:
            filtered = fourier_filter_cupy_2D(img,low_pass,cutoff)
            filtered_data.append(filtered)
            """
        #out_data = np.stack(filtered_data)
        #return out_data
    elif data.ndim > 3:
        filtered_data = []
        shape_ND = data.shape
        data = data.reshape(-1,data.shape[-2],data.shape[-1]) # This assumes N-dimensional data where the final two dimensions are reserved for ...,H,W

        num_chunks = data.shape[0] // chunk_size + int(data.shape[0] % chunk_size != 0)
        filtered_data = []
        for i in tqdm(range(num_chunks),desc="Processing Chunks"):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, data.shape[0])
            chunk = cp.asarray(data[start_idx:end_idx])
            filtered_chunk = cp.stack([_fourier_filter_cupy_2d(chunk[j], low_pass, cutoff) for j in range(chunk.shape[0])], axis=0)
            filtered_data.append(cp.asnumpy(filtered_chunk))
        if chunk_size > 1:
            out_data = np.concatenate(filtered_data,axis=0)
        elif chunk_size == 1:
            out_data = np.stack(filtered_data,axis=0)
        else:
            raise ValueError("Cannot have chunksize < 1")
        """
        for img in data:
            filtered = fourier_filter_cupy_2D(img,low_pass,cutoff)
            filtered_data.append(filtered)
        out_data = np.stack(filtered_data)
        """
        out_data = out_data.reshape(shape_ND)
    print(f"cupy used memory: {cp.get_default_memory_pool().used_bytes()} bytes\n")
    return out_data

def _fourier_filter_cupy_2d(data: ImageData, low_pass: bool = True, cutoff: float = 0.1) -> ImageData:
    #data = cp.asarray(data)
    f_transform = fftshift(fft2(data))
    rows, cols = data.shape
    crow, ccol = rows // 2, cols // 2
    mask = cp.zeros((rows, cols), dtype=cp.uint8)
    cutoff = int(cutoff * crow)
    mask = cv2.circle(cp.asnumpy(mask), (crow, ccol), cutoff, 1, -1)
    mask = cp.asarray(mask)
    
    if low_pass:
        f_transform = f_transform * mask
    else:
        f_transform = f_transform * (1 - mask)
    
    f_transform = fftshift(f_transform)
    filtered_data = ifft2(f_transform).real
    return filtered_data #cp.asnumpy(filtered_data)

@thread_worker(connect={"returned": lambda x: viewer.add_image(x, name="Filtered Image")})
def fourier_filter_worker_cupy(data: ImageData, low_pass: bool = True, cutoff: float = 0.1,chunk_size: int = 16) -> Generator[ImageData,ImageData,ImageData]:
    show_info("Filtering image with Fourier filter using CuPy")
    start = time.time()
    output = fourier_filter_cupy(data, low_pass, cutoff, chunk_size)
    processing_time = time.time() - start
    show_info(f"Image processing time: {processing_time:.4f} seconds")
    print(f"cupy used memory(thread level): {cp.get_default_memory_pool().used_bytes()} bytes\n")
    cp.get_default_memory_pool().free_all_blocks()
    print(f"cupy used memory(thread level2): {cp.get_default_memory_pool().used_bytes()} bytes\n")
    
    #memory_stats()
    return output






    
