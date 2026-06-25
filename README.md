# Zyxin Traction Vector Prediction

This repository documents a fine-tuning workflow for predicting traction force vector fields from Zyxin fluorescence images.

The current model predicts a 2-channel traction vector field:

- `Fx`
- `Fy`

The traction magnitude is calculated from the predicted Fx/Fy vector field.

## Project status

This repository is currently used as a documentation and demonstration repository.

Fine-tuned model weights are **not publicly included** until redistribution permission for the original pretrained model is confirmed.

## Input and output

### 1-channel model

Input:

```text
[Zyxin image]
