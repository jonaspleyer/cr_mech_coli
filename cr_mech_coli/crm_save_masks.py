import argparse
import numpy as np
from PIL import Image
import cr_mech_coli as crm
from pathlib import Path


def crm_save_masks_main():
    parser = argparse.ArgumentParser(
        description="Fits the Bacterial Rods model to a system of cells."
    )
    parser.add_argument("masks", nargs="+", help="List of masks")
    parser.add_argument(
        "--delim", type=str, default=",", help="Delimiter for stored masks"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=None, help="Output directory"
    )
    pyargs = parser.parse_args()

    mask_paths = pyargs.masks

    # Load all masks
    masks = [np.loadtxt(m, delimiter=pyargs.delim) for m in mask_paths]

    # Convert and store masks
    for path, m in zip(mask_paths, masks):
        m = m.astype(int)
        counters = np.unique(m)
        counters = counters[counters != 0]  # Remove background

        # Convert color
        color_mapping = {i: crm.counter_to_color(i) for i in counters}
        new_mask = np.zeros((*m.shape, 3), dtype=np.uint8)

        for counter, color in color_mapping.items():
            new_mask[m == counter] = color

        img = Image.fromarray(new_mask, mode="RGB")
        name = Path(path).stem
        opath = Path(path).parent if pyargs.output_dir is None else pyargs.output_dir
        img.save(str(Path(opath) / (name + ".png")))
        img.save(str(Path(opath) / (name + ".pdf")))
