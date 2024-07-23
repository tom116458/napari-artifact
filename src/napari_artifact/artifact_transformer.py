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
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed



class FourierFilterTorch(nn.Module):
    def __init__(self, low_pass: bool = True, cutoff: float = 0.1):
        super(FourierFilterTorch, self).__init__()
        self.low_pass = low_pass
        self.cutoff = cutoff

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        f_transform = fftshift(fft2(data))
        rows, cols = data.shape[-2:]
        crow, ccol = rows // 2, cols // 2
        mask = torch.zeros((rows, cols), dtype=torch.uint8, device=data.device)
        cutoff = int(self.cutoff * crow)
        mask_np = cv2.circle(mask.cpu().numpy(), (crow, ccol), cutoff, 1, -1)
        mask = torch.tensor(mask_np, device=data.device)

        if self.low_pass:
            f_transform = f_transform * mask
        else:
            f_transform = f_transform * (1 - mask)

        f_transform = fftshift(f_transform)
        filtered_data = ifft2(f_transform).real
        return filtered_data

def process_chunk(model, chunk, idx):
    with torch.no_grad():
        chunk = torch.tensor(chunk, dtype=torch.float32).to(torch.device("cuda"))
        filtered_chunk = model(chunk).cpu().numpy()
    return idx, filtered_chunk

def fourier_filter_torch(data: ImageData, low_pass: bool = True, cutoff: float = 0.1, chunk_size: int = 16) -> ImageData:
    print(f"torch used memory(start): {torch.cuda.memory_allocated()}\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FourierFilterTorch(low_pass, cutoff).to(device)
    if data.ndim == 2:
        model_data = torch.tensor(data, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            out_data = model(model_data).cpu().numpy()

    elif data.ndim == 3:
        num_chunks = data.shape[0] // chunk_size + int(data.shape[0] % chunk_size != 0)
        filtered_data = [None] * num_chunks

        with ThreadPoolExecutor() as executor:
            futures = []
            for i in tqdm(range(num_chunks), desc="Processing Chunks"):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, data.shape[0])
                chunk = data[start_idx:end_idx].copy()
                futures.append(executor.submit(process_chunk, model, chunk, i))

            for future in as_completed(futures):
                idx, result = future.result()
                filtered_data[idx] = result

        out_data = np.concatenate(filtered_data, axis=0)

    elif data.ndim > 3:
        shape_ND = data.shape
        data = data.reshape(-1, data.shape[-2], data.shape[-1])
        num_chunks = data.shape[0] // chunk_size + int(data.shape[0] % chunk_size != 0)
        
        filtered_data = [None] * num_chunks

        with ThreadPoolExecutor() as executor:
            futures = []
            for i in tqdm(range(num_chunks), desc="Processing Chunks"):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, data.shape[0])
                chunk = data[start_idx:end_idx].copy()
                futures.append(executor.submit(process_chunk, model, chunk, i))

            for future in as_completed(futures):
                idx, result = future.result()
                filtered_data[idx] = result

        out_data = np.concatenate(filtered_data, axis=0).reshape(shape_ND)

    print(f"torch used memory(end): {torch.cuda.memory_allocated()}\n")
    return out_data

@magic_factory(low_pass={"label": "Low Pass Filter", "widget_type": "CheckBox"}, cutoff={"label": "Cutoff Frequency", "widget_type": "FloatSlider", "min": 0.01, "max": 0.5, "step": 0.01})
def fourier_filter_gui_torch(img: Image, low_pass: bool = True, cutoff: float = 0.1, chunk_size: int = 16):
    show_info("Filtering image with Fourier filter using nn model")
    start = time.time()
    output = fourier_filter_torch(img.data, low_pass, cutoff, chunk_size)
    processing_time = time.time() - start
    show_info(f"Image processing time: {processing_time:.4f} seconds")
    print(f"torch used memory(start): {torch.cuda.memory_allocated()}\n")
    torch.cuda.empty_cache()
    print(f"torch used memory(end): {torch.cuda.memory_allocated()}\n")
    viewer.add_image(output, name="Filtered Image")
    
