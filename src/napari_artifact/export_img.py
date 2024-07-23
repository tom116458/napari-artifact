import os
import matplotlib.pyplot as plt
import napari
from magicgui import magic_factory
from napari_cool_tools_io import viewer, memory_stats
from napari.qt.threading import thread_worker
from napari.layers import Image
from napari.types import ImageData
 
@magic_factory(call_button="Save PNGs")
def save_labels_as_png_gui(img: Image,viewer=None, save_path=r"C:\Users\Team_ROP\Documents\haoshen_qin\for_spencer\Segmentations"):
    if viewer is None:
        viewer = napari.current_viewer()
    save_labels_as_png(viewer, save_path, img)


# Function to save all label layers in a Napari viewer as PNGs
def save_labels_as_png(viewer, save_path, img: Image):
    """
    Save all label layers from a Napari viewer as PNG files.

    Parameters:
    - viewer: napari.Viewer, an instance of a Napari viewer.
    - save_path: str, the path to the folder where PNG files will be saved.
    """
    # Ensure the save path exists
    os.makedirs(save_path, exist_ok=True)

    # Iterate through all layers in the viewer

    for layer in viewer.layers:
        # Check if the layer is a labels layer
        if isinstance(layer, napari.layers.Labels):
            # Construct the file path
            file_path = os.path.join(save_path, f"{img.name}.png")
            # Save the layer as a PNG file
            plt.imsave(file_path, layer.data, cmap='gray')
            print(f"Saved {file_path}")
     
