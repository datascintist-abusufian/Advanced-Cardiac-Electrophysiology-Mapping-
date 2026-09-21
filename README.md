# Advanced Cardiac Electrophysiology Mapping

Automated signal windowing, spatial interpolation and parallel processing for optical mapping of cardiac electrophysiology.

## Background

Optical mapping records membrane-potential or calcium transients across the surface of a heart preparation as a stack of images: two spatial dimensions and one time dimension. A single recording holds thousands of frames, and pulling activation and repolarisation information out of every pixel by hand is the bottleneck in most laboratories. This project automates the windowing of those signals and parallelises the per-pixel work so that a full intervention series can be processed consistently rather than one file at a time.

## Data

Recordings are MATLAB `.mat` files, each containing an `images` variable of shape 51 x 51 x 5000, that is a 51 by 51 pixel field of view sampled over 5000 frames.

The reference dataset is a mouse ventricular preparation covering four experimental axes.

- Rhythm: paced and sinus rhythm
- Pharmacology: flecainide at baseline and at 1, 2 and 3 micromolar
- Ischaemia: low-flow baseline and 1 to 6 minutes of low flow
- Temperature: cooled preparation, paced and in sinus rhythm

The recordings themselves are not redistributed in this repository. Please get in touch about access.

## What the notebook does

`Additional_Analysis.ipynb` walks through the analysis end to end.

1. Unpacks the recording archive and enumerates the available `.mat` files, normalising folder names that contain spaces.
2. Loads each recording with `scipy.io.loadmat`, verifies the `images` key and reports the array shape.
3. Extracts individual frames and renders them for visual quality control before any quantitative step.
4. Applies ordinary kriging with PyKrige to interpolate across the pixel grid, producing smooth spatial maps from a noisy sensor array.
5. Carries the windowed per-pixel signals into downstream statistical comparison across the pharmacological, ischaemic and temperature conditions listed above.

## Stack

Python 3.11 with NumPy, SciPy, Matplotlib, scikit-learn and PyKrige. Developed in Google Colab on an A100 runtime; a GPU is not required for the analysis itself.

```bash
pip install numpy scipy matplotlib scikit-learn pykrige
```

## Status

This is the working analysis notebook behind the method, not yet a packaged library. The planned next steps are to refactor the windowing and kriging steps into importable modules, add a command-line entry point for batch processing of a recording directory, and add tests against a small synthetic recording.

## Related projects

- [Deep-Spatiotemporal-Modelling-of-Cardiomyocyte-Ageing-Dysfunction](https://github.com/datascintist-abusufian/Deep-Spatiotemporal-Modelling-of-Cardiomyocyte-Ageing-Dysfunction) - motion phenotyping of cardiomyocyte ageing
- [triFuse-pytorch](https://github.com/datascintist-abusufian/triFuse-pytorch) - scribble-supervised cardiac MRI segmentation
- [CardioMap-Pro-2.0](https://github.com/datascintist-abusufian/CardioMap-Pro-2.0) - related cardiac mapping work

## Author

Md Abu Sufian, PhD researcher, School of Architecture, Computing and Engineering, University of East London. [GitHub profile](https://github.com/datascintist-abusufian) | [LinkedIn](https://www.linkedin.com/in/tacticalbusinessintelligence/)
