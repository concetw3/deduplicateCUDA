from typing import List, Tuple, Union
import click
import numpy as np
import vpype as vp
import vpype_cli
from shapely.geometry import MultiLineString
from tqdm import tqdm
from numba import cuda

@cuda.jit
def compare_lines_gpu(line_arr, tolerance, mask):
    i = cuda.grid(1)
    if i < len(line_arr) - 1:
        for j in range(i + 1, len(line_arr)):
            if all_close(line_arr[i], line_arr[j], tolerance):
                mask[j] = True
            elif all_close(line_arr[i][::-1], line_arr[j], tolerance):  # Reversed comparison
                mask[j] = True

@cuda.jit(device=True)
def all_close(a, b, atol):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if abs(a[i, j] - b[i, j]) > atol:
                return False
    return True

def _deduplicate_layer(lines: vp.LineCollection, tolerance: float, progress_bar: bool, keep_duplicates: bool) -> Tuple[vp.LineCollection, vp.LineCollection]:
    """Deduplicate lines of a single layer."""

    # Split all lines into segments
    split_lines = vp.LineCollection()
    for line in lines:
        split_lines.extend([line[i : i + 2] for i in range(len(line) - 1) if line[i] != line[i + 1]])

    lc = vp.LineCollection()
    removed_lines = vp.LineCollection()
    line_arr = np.array([np.array(line.coords) for line in split_lines.as_mls().geoms])
    mask = np.zeros(len(line_arr), dtype=bool)

    # Move data to GPU
    d_line_arr = cuda.to_device(line_arr)
    d_mask = cuda.to_device(mask)

    # Configure the blocks
    threadsperblock = 32
    blockspergrid = (len(line_arr) + (threadsperblock - 1)) // threadsperblock

    # Run the GPU function
    compare_lines_gpu[blockspergrid, threadsperblock](d_line_arr, tolerance, d_mask)

    # Copy the result back to the host
    d_mask.copy_to_host(mask)

    if keep_duplicates:
        removed_lines.extend(MultiLineString(list(line_arr[mask])))

    line_arr = line_arr[~mask]
    lc.extend(MultiLineString(list(line_arr)))

    return lc, removed_lines

@click.command()
@click.option(
    "-t",
    "--tolerance",
    type=vpype_cli.LengthType(),
    default="0.01mm",
    help="Max distance between points to consider them equal (default: 0.01mm)",
)
@click.option(
    "-p", "--progress-bar", is_flag=True, default=True, help="(flag) Display a progress bar"
)
@click.option(
    "-l",
    "--layer",
    type=vpype_cli.LayerType(accept_multiple=True),
    default="all",
    help="Target layer(s) (default: 'all')",
)
@click.option(
    "-k",
    "--keep-duplicates",
    is_flag=True,
    default=False,
    help="(flag) Keep removed duplicates in a separate layer",
)
@vpype_cli.global_processor
def deduplicate(
    document: vp.Document,
    tolerance: float,
    progress_bar: bool,
    layer: Union[int, List[int]],
    keep_duplicates: bool,
) -> vp.Document:
    """Remove duplicate lines."""

    layer_ids = vpype_cli.multiple_to_layer_ids(layer, document)
    new_document = document.empty_copy()
    removed_layer_id = document.free_id()

    for lines, l_id in zip(document.layers_from_ids(layer_ids), layer_ids):
        new_lines, removed_lines = _deduplicate_layer(lines, tolerance, progress_bar, keep_duplicates)
        new_document.add(new_lines, layer_id=l_id)

        if keep_duplicates and not removed_lines.is_empty():
            new_document.add(removed_lines, layer_id=removed_layer_id)

    return new_document

deduplicate.help_group = "Plugins"
