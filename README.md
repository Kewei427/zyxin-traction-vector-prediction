# Zyxin-to-Traction Force Prediction

This repository documents the process of adapting a deep learning model to predict cellular traction force fields from Zyxin fluorescence images.

The project focuses on predicting traction force information from single-cell Zyxin images. The adapted pipeline predicts a traction vector field with two components:

* `Fx`: x-direction traction component
* `Fy`: y-direction traction component

The traction magnitude map is then calculated from the predicted Fx/Fy components.

## Project background

Traction force microscopy is used to measure the mechanical forces exerted by cells on a substrate. In this project, Zyxin fluorescence images are used as the input signal for predicting traction force patterns.

The work is based on adapting a pretrained traction-force prediction model to a new Zyxin dataset. The goal is to make the model work with the current data format, fine-tune it on the available samples, and evaluate how accurately it predicts traction force maps.

## Data format

For each cell, the adapted pipeline uses:

* Zyxin fluorescence image,
* cell mask,
* traction vector label,
* metadata from the traction force microscopy reconstruction.

The numerical traction label is stored as:

```text
traction_vector.npy
```

This file contains two channels:

```text
channel 0 = Fx
channel 1 = Fy
```

The PNG vector images are used only for visualization and are not used as training labels.

## Model target

The model predicts a two-channel vector field:

```text
[Fx, Fy]
```

The traction magnitude is calculated after prediction:

```text
magnitude = sqrt(Fx^2 + Fy^2)
```

No sigmoid activation is used on the model output because Fx and Fy can contain both positive and negative values.

## Preprocessing process

The preprocessing workflow was modified for the new Zyxin dataset.

For each cell, the pipeline:

1. Loads the Zyxin fluorescence image.
2. Loads the cell mask.
3. Loads the traction vector label from `traction_vector.npy`.
4. Extracts Fx and Fy from the vector label.
5. Normalizes the Zyxin image to the range 0–1.
6. Converts the cell mask into a binary mask.
7. Scales Fx and Fy using the selected force scale.
8. Crops or pads all arrays to 512 × 512.
9. Converts the processed arrays into tensors for model training.

The final tensor shapes are:

```text
input:  [batch, channels, 512, 512]
target: [batch, 2, 512, 512]
mask:   [batch, 1, 512, 512]
```

## Vector-field training process

The original magnitude-only direction was replaced with Fx/Fy vector-field prediction.

This change was made because predicting Fx and Fy preserves both force magnitude and direction information. It also makes the fine-tuning task more consistent with the pretrained model structure.

The model output was changed to two channels:

```text
output channel 0 = Fx
output channel 1 = Fy
```

After prediction, the traction magnitude is calculated from Fx and Fy for visualization and evaluation.

## Data augmentation

Vector-aware flipping was added during training.

Because the target is a vector field, the vector direction must be corrected when the image is flipped.

For horizontal flip:

```text
Fx changes sign
Fy stays the same
```

For vertical flip:

```text
Fy changes sign
Fx stays the same
```

This keeps the augmented vector labels physically consistent.

## Loss function

The training loss was modified for vector-field prediction.

The loss combines:

* pixel-level Fx/Fy mean squared error,
* pixel-level Fx/Fy L1 error,
* traction magnitude error,
* cell-level mean magnitude error,
* light smoothness loss.

This loss is designed to evaluate both the spatial vector-field prediction and the whole-cell traction-level prediction.

## Zyxin-only model

The first adapted model used only the Zyxin image as input.

Input:

```text
Zyxin image
```

Output:

```text
Fx, Fy
```

This model successfully trained and produced traction vector-field predictions.

Validation results:

| Metric                        |  Value |
| ----------------------------- | -----: |
| Mean vector MAE               | 453.21 |
| Mean magnitude MAE            | 368.32 |
| Cell-level mean magnitude MAE | 135.03 |
| Mean relative error           | 28.73% |

This showed that the vector-field pipeline was working, but some predictions still showed overestimation in low-traction cells and noisy spatial maps.

## Zyxin + cell mask model

A second model setting was tested by adding the cell mask as an additional input channel.

Input:

```text
Zyxin image + cell mask
```

Output:

```text
Fx, Fy
```

The cell mask provides cell boundary and morphology information. This helps the model identify the cell region and estimate the whole-cell traction level more accurately.

Because the pretrained model did not directly support a simple two-channel input, a small 1 × 1 input adapter was added before the base model.

The adapter maps:

```text
[Zyxin, mask] -> adapted input
```

The adapter was initialized so that the model initially behaves like the Zyxin-only model:

```text
Zyxin channel weight = 1
mask channel weight = 0
```

During fine-tuning, the adapter learns whether the mask channel is useful.

Validation results:

| Metric                        |  Value |
| ----------------------------- | -----: |
| Mean vector MAE               | 446.30 |
| Mean magnitude MAE            | 361.99 |
| Cell-level mean magnitude MAE | 108.13 |
| Mean relative error           | 22.30% |

## Comparison between model settings

| Model setting     | Vector MAE | Magnitude MAE | Cell-level Mean MAE | Mean Relative Error |
| ----------------- | ---------: | ------------: | ------------------: | ------------------: |
| Zyxin only        |     453.21 |        368.32 |              135.03 |              28.73% |
| Zyxin + cell mask |     446.30 |        361.99 |              108.13 |              22.30% |

Adding the cell mask improved all four validation metrics.

The largest improvement was in whole-cell traction-level prediction. The cell-level mean magnitude MAE decreased from 135.03 to 108.13, and the mean relative error decreased from 28.73% to 22.30%.

This suggests that cell boundary and morphology information help the model estimate the overall traction level more accurately.

## Whole-map traction force evaluation

In addition to pixel-level metrics, a whole-map traction-force evaluation was added.

For each validation cell:

1. Fx and Fy are used to calculate the traction magnitude map.
2. The cell mask is applied.
3. The traction magnitude is summed inside the cell mask.
4. The predicted total traction is compared with the true total traction.

The evaluation reports:

* true total traction,
* predicted total traction,
* absolute total traction error,
* relative total traction error.

This metric measures how different the predicted whole-cell traction map is from the ground truth.

If strict physical total force is required, the summed traction magnitude can be multiplied by the pixel area from the metadata. For relative error comparison, the pixel area does not change the percentage error because it is applied to both the true and predicted maps.

## Visualization process

Visualization was added to inspect the model prediction qualitatively.

For each validation cell, the visualization includes:

* Zyxin input image,
* cell mask,
* true traction vector field,
* predicted traction vector field,
* magnitude error map.

The visualizations showed that the model can predict non-zero traction patterns inside the cell region and capture some overall force structure.

However, the predicted maps are still less spatially localized than the ground truth in some cases. Some predictions also remain noisy, suggesting that spatial precision still depends on data quality, dataset size, and TFM reconstruction resolution.

## Summary of modifications

The main modifications made in this project are:

1. Changed the task from magnitude-only prediction to Fx/Fy vector-field prediction.
2. Built a new dataset pipeline for Zyxin images, cell masks, and traction vector labels.
3. Added vector-aware data augmentation.
4. Modified the loss function for vector-field training.
5. Added cell-level mean traction evaluation.
6. Added whole-map total traction evaluation.
7. Tested the effect of adding cell mask information as an additional input.
8. Added validation visualizations for traction vector-field comparison.

## Current conclusion

The adapted Zyxin-to-traction pipeline can predict traction vector fields from Zyxin fluorescence images.

The Zyxin + cell mask model produced better validation results than the Zyxin-only model in the current validation split.

The main improvement is in whole-cell traction-level estimation, where the mean relative error decreased from 28.73% to 22.30%.

The spatial force-map prediction still has limitations, especially in terms of noise and localization of traction hotspots.

## License and redistribution note

This project adapts a pretrained model from the public `schmittms/cell_force_prediction` repository.

At the time of checking, no explicit license file was visible in the original repository. Therefore, fine-tuned model weights are not publicly redistributed in this repository unless redistribution permission is confirmed.

This repository is currently used to document the adaptation process, fine-tuning changes, evaluation results, and visualization examples.
