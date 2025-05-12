from pathlib import Path
import requests
import subprocess
from glob import glob
import cv2 as cv
import numpy as np
import cr_mech_coli as crm
import skimage as sk
from tqdm import tqdm
import multiprocessing as mp

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


def extract_mask(iteration, img, save_progression: bool = False):
    img2 = np.copy(img)
    filt1 = img2[:, :, 1] <= 150
    img2[filt1] = [0, 0, 0]
    filt2 = np.all(img2 >= np.array([180, 180, 180]), axis=2)
    img2[filt2] = [0, 0, 0]

    cutoff = int(img2.shape[1] / 3)
    filt3 = np.linalg.norm(img2 - GREEN_COLOR, axis=2) >= 100
    filt3[:, :cutoff] = True
    img2[filt3] = [0, 0, 0]

    img3 = np.copy(img2)
    img3[filt3 == 0] = GREEN_COLOR.astype(np.uint8)

    img_filt = sk.segmentation.expand_labels(img3, distance=20)

    img3 = np.repeat(np.all(img_filt != [0, 0, 0], axis=2), 3).reshape(
        img_filt.shape
    ).astype(int) * GREEN_COLOR.astype(int)
    img4 = np.copy(img3).astype(np.uint8)

    try:
        pos = crm.extract_positions(img4)[0][0]
        p = pos[:, ::-1].reshape((-1, 1, 2))
        img4 = cv.polylines(
            np.copy(img),
            [p.astype(int)],
            isClosed=False,
            color=(10, 10, 230),
            thickness=2,
        )
        ret = pos
    except ValueError as e:
        print(e)
        ret = None

    if save_progression:
        cv.imwrite(f"progression-{iteration:06}-1.png", img)
        cv.imwrite(f"progression-{iteration:06}-2.png", img2)
        cv.imwrite(f"progression-{iteration:06}-3.png", img3)
        cv.imwrite(f"progression-{iteration:06}-4.png", img4)

    return ret


def extract_masks(path: Path, save_progressions: list[int] = []):
    files = sorted(glob(str(path / "frames/*")))
    imgs = [cv.imread(f) for f in files]

    for img_file, img in tqdm(zip(files, imgs), total=len(imgs)):
        it = Path(img_file).stem
        it = int(it)
        position = extract_mask(it, img, it in save_progressions)
        if position is not None and np.sum(position.shape) > 0:
            np.savetxt((path / "positions") / f"position-{it:06}.txt", position)
        else:
            print(f"[{it:06}] Could not extract positions")


if __name__ == "__main__":
    for name, url in zip(NAMES, LINKS):
        path, filename = download_movie(url, name)
        # get_frames(path, filename)
        extract_masks(path, save_progressions=[32])
        exit()
