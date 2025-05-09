from pathlib import Path
import requests
import subprocess
from glob import glob
import cv2 as cv
import numpy as np
import cr_mech_coli as crm
import skimage as sk
import sklearn as skl

MOVIE1 = "https://www.pnas.org/doi/suppl/10.1073/pnas.1317497111/suppl_file/sm01.mov"
MOVIE2 = "https://www.pnas.org/doi/suppl/10.1073/pnas.1317497111/suppl_file/sm02.mov"

LINKS = [MOVIE1, MOVIE2]
NAMES = ["elastic", "plastic"]

GREEN_COLOR = np.array([21.5 / 100, 86.6 / 100, 21.6 / 100]) * 255


def download_movie(url, name):
    filename = url.split("/")[-1]
    p = Path(name)

    response = requests.get(url)
    if response.status_code == 200:
        p.mkdir(exist_ok=True, parents=True)
        with open(p / filename, "wb") as file:
            file.write(response.content)
    else:
        print(f"Could not download link: {url}")
        print(
            "If you have already downloaded the files manually, this script will continue working as expected."
        )
    return p, filename


def get_frames(path, filename):
    # Execute ffmpeg
    frame_dir = path / "frames"
    frame_dir.mkdir(exist_ok=True, parents=True)
    cmd = f"ffmpeg -loglevel quiet -i {path / filename} {frame_dir / '%06d.png'}"
    subprocess.Popen(cmd, text=True, shell=True)


def extract_masks(path):
    files = sorted(glob(str(path / "frames/*")))
    imgs = [cv.imread(f) for f in files]

    img = imgs[10]

    img1 = np.array(img)
    filt1 = img[:, :, 1] <= 150
    img1[filt1] = [0, 0, 0]
    filt2 = np.all(img1 >= np.array([180, 180, 180]), axis=2)
    img1[filt2] = [0, 0, 0]

    filt3 = np.linalg.norm(img1 - GREEN_COLOR, axis=2) >= 50
    img1[filt3] = [0, 0, 0]

    img_filt = sk.segmentation.expand_labels(img1, distance=10)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, figsize=(6, 8))

    ax[0, 0].imshow(img1)
    ax[0, 1].imshow(img_filt)
    img3 = np.all(img_filt != [0, 0, 0], axis=2)
    ax[1, 0].imshow(img3)
    seed = np.copy(img3)
    seed[1:-1, 1:-1] = img3.max()
    filled = sk.morphology.reconstruction(seed, img3, method="erosion")
    ax[1, 1].imshow(filled)

    for i in range(2):
        for j in range(2):
            ax[i, j].axis("off")

    try:
        positions = crm.extract_positions(img_filt)[0]
        ax[0, 1].plot(positions[0, :, 1], positions[0, :, 0], color="red", linewidth=2)
    except ValueError as e:
        print("Could not extract positions")
        print(e)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    for name, url in zip(NAMES, LINKS):
        path, filename = download_movie(url, name)
        # get_frames(path, filename)
        extract_masks(path)
        exit()
