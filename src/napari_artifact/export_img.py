import os
import numpy as np
import napari
from magicgui import magic_factory
from napari.layers import Layer, Image, Labels
import imageio
from napari.layers import Layer

@magic_factory(call_button="Save PNG")
def save_slices_as_png_gui(
    img: Layer,
    save_path: str = r"C:\Haoshen\Work\test\normal",
    start_slice: int = 0,
    end_slice: int = 10,
    viewer: napari.Viewer = None,
):
    if viewer is None:
        viewer = napari.current_viewer()
    save_slices_as_png(img, save_path, start_slice, end_slice)

def save_slices_as_png(
    img: Layer,
    save_path: str,
    start_slice: int,
    end_slice: int,
):
    # Determine the save folder based on the layer type
    if isinstance(img, Image):
        save_folder = os.path.join(save_path, 'images')
    elif isinstance(img, Labels):
        save_folder = os.path.join(save_path, 'labels')
    else:
        print("Unsupported layer type.")
        return

    # Create the folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    data = img.data

    if data.ndim != 3:
        print("Only 3D data is supported.")
        return

    # Ensure slice indices are within the valid range
    start_slice = max(0, start_slice)
    end_slice = min(data.shape[0] - 1, end_slice)

    for i in range(start_slice, end_slice + 1):
        slice_data = data[i]

        if isinstance(img, Image):
            # For Image layers, adjust contrast
            vmin, vmax = img.contrast_limits
            if vmax == vmin:
                print("Max and min are the same, cannot normalize.")
                return
            adjusted_slice = np.clip(slice_data, vmin, vmax)
            adjusted_slice = (adjusted_slice - vmin) / (vmax - vmin)
            adjusted_slice_uint8 = (adjusted_slice * 255).astype(np.uint8)
            # Use the actual slice index in the filename
            filename = f"{img.name}_{i}.png"
            file_path = os.path.join(save_folder, filename)
            imageio.imwrite(file_path, adjusted_slice_uint8)
        elif isinstance(img, Labels):
            # For Labels layers, normalize to 0-255 for visibility
            slice_data = slice_data.astype(np.float32)
            max_label = slice_data.max()
            if max_label > 0:
                adjusted_slice = (slice_data / max_label) * 255
            else:
                adjusted_slice = slice_data
            adjusted_slice_uint8 = adjusted_slice.astype(np.uint8)
            # Use the actual slice index in the filename
            filename = f"{img.name}_{i}_label.png"
            file_path = os.path.join(save_folder, filename)
            imageio.imwrite(file_path, adjusted_slice_uint8)
        else:
            print("Unsupported layer type.")
            return

        print(f"Saved {file_path}")
