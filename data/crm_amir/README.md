# Bending forces plastically deform growing bacterial cell walls

## Abstract
Cell walls define a cell’s shape in bacteria.
The walls are rigid to resist large internal pressures, but remarkably plastic to adapt to a wide
range of external forces and geometric constraints.
Currently, it is unknown how bacteria maintain their shape.
In this paper, we develop experimental and theoretical approaches and show that mechanical stresses
regulate bacterial cell wall growth.
By applying a precisely controllable hydrodynamic force to growing rod-shaped Escherichia coli and
Bacillus subtilis cells, we demonstrate that the cells can exhibit two fundamentally different
modes of deformation.
The cells behave like elastic rods when subjected to transient forces, but deform plastically when
significant cell wall synthesis occurs while the force is applied.
The deformed cells always recover their shape.
The experimental results are in quantitative agreement with the predictions of the theory of
dislocation-mediated growth.
In particular, we find that a single dimensionless parameter, which depends on a combination of
independently measured physical properties of the cell, can describe the cell’s responses under
various experimental conditions.
These findings provide insight into how living cells robustly maintain their shape under varying
physical environments.

| | |
|:---|---|
| Link | https://doi.org/10.1073/pnas.1317497111 |
| DOI | [https://doi.org/10.1073/pnas.1317497111](https://doi.org/10.1073/pnas.1317497111) |
| License | [CC BY-NC-ND or CC BY](https://creativecommons.org/licenses/by/4.0/deed.en) |
| Movie 1 | https://www.pnas.org/doi/suppl/10.1073/pnas.1317497111/suppl_file/sm01.mov |
| Movie 2 | https://www.pnas.org/doi/suppl/10.1073/pnas.1317497111/suppl_file/sm02.mov |

## Obtaining the Data
We provide a small python script `obtain.py` in order to obtain the movie, extract images and sort
them into subfolders.
Notice that this script might not work from your location and you might need to manually download
the files before you can proceed.
Make sure to place them inside the folders `elastic` and `plastic` respectively.
Afterwards, the script can be run again to produce the frames with `ffmpeg`.

```bash
python3 obtain.py
```
