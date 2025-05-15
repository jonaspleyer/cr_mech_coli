# Bacillus subtilis (Youtube)

## Description

| | |
|:---|---|
| Link | [www.youtube.com/watch?v=hUt55R8qU9g](https://www.youtube.com/watch?v=hUt55R8qU9g) |
| License | [Youtube standard license](https://www.youtube.com/static?template=terms) |

## Obtaining the Data
We provide a small bash script `obtain.sh` in order to obtain the movie, extract images and sort
them into subfolders.
The script requires `wget` and `ffmpeg` to be installed.

```bash
./obtain.sh
```

## Segmenting the Cells

The jupyter notebook `segment.ipynb` is used to segment the cells.
Note that we installed omnipose for `python3.10` from the submodule at the base of the repository.
From the root of the repositroy perform the following actions.

```bash
cd omnipose
uv pip install -e .
```