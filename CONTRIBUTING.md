# Contributing
## PR Process
1. Explain the problem your are trying to solve/the feature your are adding
    - Your commit history should be clean and explanatory
    - Avoid large bulk commits, separate them by intent and effect
1. Remember to update documentation if required
    - Update stub files `*.pyi` when changing parts of the Rust code
    - Run `pytest` and `cargo teest` to ensure compatibility across the package
    - Suggest version bumping depending on the changes made. (See semver.org).
2. Do not include any large amounts of data files
    - instead write scripts to automatically obtain them (most likely download)
    - See `data/..` directories for examples

## 💡 Ask Questions
Have a look at the [documentation](https://jonaspleyer.github.io/cr_mech_coli/).
If you still have any questions, feel free to open a new
[discussion](https://github.com/jonaspleyer/cr_mech_coli/discussions).

## 🪲 Bug Reports
Create a [new issue](https://github.com/jonaspleyer/cr_mech_coli/issues).
If you think that the bug might stem from
[`cellular_raza`](https://github.com/jonaspleyer/cellular_raza), head over there

## 💌 Feature Requests
If you would like to see any particular feature, consider starting a
[discussion](#bulb-ask-questions).
If you already have a design, feel free to open a [PR](#pr-process).
