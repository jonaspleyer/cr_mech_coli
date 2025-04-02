import requests
from pathlib import Path
import subprocess
import tqdm

# The commented videos should not be used as source material
DOWNLOAD_LINKS = [
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-03446-y/MediaObjects/41467_2018_3446_MOESM4_ESM.avi",
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-03446-y/MediaObjects/41467_2018_3446_MOESM5_ESM.avi",
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-03446-y/MediaObjects/41467_2018_3446_MOESM6_ESM.mp4",
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-03446-y/MediaObjects/41467_2018_3446_MOESM6_ESM.mp4",
    # "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-03446-y/MediaObjects/41467_2018_3446_MOESM7_ESM.mp4",
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-03446-y/MediaObjects/41467_2018_3446_MOESM8_ESM.mp4",
    # "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-03446-y/MediaObjects/41467_2018_3446_MOESM9_ESM.mp4",
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-03446-y/MediaObjects/41467_2018_3446_MOESM10_ESM.mp4",
    # "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-03446-y/MediaObjects/41467_2018_3446_MOESM11_ESM.mp4",
    # "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-03446-y/MediaObjects/41467_2018_3446_MOESM12_ESM.mp4",
]

for n, url in tqdm.tqdm(enumerate(DOWNLOAD_LINKS), total=len(DOWNLOAD_LINKS)):
    filename = url.split("/")[-1]
    p = Path(filename.split(".")[0])

    response = requests.get(url)
    # subprocess.run(f"wget {filename}", text=True, shell=True)
    if response.status_code == 200:
        p.mkdir(exist_ok=True, parents=True)
        with open(p / filename, "wb") as file:
            file.write(response.content)

        # Execute ffmpeg
        frame_dir = p / "frames"
        frame_dir.mkdir(exist_ok=True, parents=True)
        cmd = f"ffmpeg -loglevel quiet -i {p / filename} {frame_dir / '%06d.png'}"
        subprocess.Popen(cmd, text=True, shell=True)
