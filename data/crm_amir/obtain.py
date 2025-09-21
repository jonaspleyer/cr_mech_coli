from pathlib import Path
import requests
import subprocess
from glob import glob
import cv2 as cv
import numpy as np
import cr_mech_coli as crm
from tqdm import tqdm

MOVIE1 = "https://www.pnas.org/doi/suppl/10.1073/pnas.1317497111/suppl_file/sm01.mov"
MOVIE2 = "https://www.pnas.org/doi/suppl/10.1073/pnas.1317497111/suppl_file/sm02.mov"

LINKS = [MOVIE1, MOVIE2]
NAMES = ["elastic", "plastic"]


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
