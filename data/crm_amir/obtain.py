from tqdm import tqdm
from pathlib import Path
import requests
import subprocess

MOVIE1 = "https://www.pnas.org/doi/suppl/10.1073/pnas.1317497111/suppl_file/sm01.mov"
MOVIE2 = "https://www.pnas.org/doi/suppl/10.1073/pnas.1317497111/suppl_file/sm02.mov"

LINKS = [MOVIE1, MOVIE2]
NAMES = ["elastic", "plastic"]

for n, url in tqdm(enumerate(LINKS), total=len(LINKS)):
    filename = url.split("/")[-1]
    p = Path(NAMES[n])

    response = requests.get(url)
    if response.status_code == 200:
        p.mkdir(exist_ok=True, parents=True)
        with open(p / filename, "wb") as file:
            file.write(response.content)
    else:
        print(f"Could not download link: {url}")
        print(
            "If you have already downloaded the files manually,\
            this script will continue working as expected."
        )

    # Execute ffmpeg
    frame_dir = p / "frames"
    frame_dir.mkdir(exist_ok=True, parents=True)
    cmd = f"ffmpeg -loglevel quiet -i {p / filename} {frame_dir / '%06d.png'}"
    subprocess.Popen(cmd, text=True, shell=True)
