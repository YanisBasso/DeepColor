# DeepColor
A Deep Learning Framework for image color calibration

# Introduction

The colors displayed in an image are not only dependent on the observed surfaces (objects), but also on the lighting conditions during the acquisition, on the sensitivities of the camera sensors, on the camera settings, ... However, for most of the computer vision applications (feature extraction, classification, tracking, ...), the only important features are those of the surfaces while the other parameters are considered as noise and disturb the results.

This project addresses this issue by proposing a solution for a color auto-calibration solution for cameras, such that any acquisition device is able to normalize its colors and render them invariant to the acquisition conditions. This solution will be based on recent methods taken from the artificial intelligence area. Our new approach consists in exploiting a set of color images containing a color test pattern with a set of calibrated color patches. The very idea is to mimic recent advent
`
