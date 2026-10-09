# Multi-Task Retinal Fundus Image Analysis

A deep learning project exploring whether retinal fundus images can be used to predict **patient gender, age, and diabetes status** through a shared multi-task model. The project uses the ODIR-5K Ocular Disease Recognition dataset and an EfficientNet-B0 backbone enhanced with a lightweight spatial-attention block.

> **Important:** This is an educational research project, not a medical diagnostic tool. Predictions should not be used for clinical decisions. Reported metrics should be interpreted cautiously until evaluated on a separate, patient-independent test set.

## Project Overview

The notebook builds a multi-task neural network with three output heads:

- **Gender classification** — binary prediction from the image.
- **Age estimation** — regression, with age normalized during training and converted back to years for evaluation.
- **Diabetes classification** — binary prediction using the dataset's `D` label.

The model shares visual features across tasks rather than training three independent networks.

## Methodology

1. **Dataset loading:** Reads the ODIR-5K `full_df.csv` metadata and training fundus images.
2. **Image preprocessing experiments:**
   - CLAHE-based contrast enhancement in LAB color space.
   - Gaussian illumination correction.
   - Approximate optic-disc cropping based on the brightest region.
3. **Input preparation:** Resizes images to 224 × 224, converts them to tensors, and applies ImageNet normalization.
4. **Feature extraction:** Uses an ImageNet-pretrained EfficientNet-B0 backbone.
5. **Spatial attention:** Combines dilated convolutions with dilation rates 2, 3, and 4 to generate a spatial attention map.
6. **Multi-task prediction:** Uses separate output heads for gender, age, and diabetes.
7. **Evaluation and interpretation:** Includes classification metrics, age-regression metrics, ROC curves, and a Grad-CAM visualization for the diabetes prediction head.

## Architecture

```text
Retinal Fundus Image
        |
Resize + Normalize
        |
EfficientNet-B0 Feature Extractor
        |
Light Spatial Attention
(dilated convolutions: rates 2, 3, 4)
        |
Global Average Pooling
        |
Shared Feature Vector
   /        |         \
Gender     Age      Diabetes
Binary   Regression   Binary
```

## Tech Stack

- Python
- PyTorch and Torchvision
- OpenCV
- NumPy and Pandas
- Pillow
- Matplotlib
- scikit-learn
- SciPy
- tqdm
- Google Colab / Google Drive

## Dataset

**ODIR-5K — Ocular Disease Intelligent Recognition**

Kaggle dataset: https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

The notebook expects the dataset metadata file `full_df.csv` and image files from the dataset's `Training Images` directory.

### Expected data layout

```text
ODIR-5K/
├── full_df.csv
└── Training Images/
    ├── ...
```

The notebook currently uses Google Drive paths. Update `csv_path` and `image_dir` to match your own environment before running it.

## How to Run

1. Download the ODIR-5K dataset from Kaggle and accept any dataset terms shown there.
2. Open `with_attention_blocks.ipynb` in Google Colab or a local Jupyter environment.
3. Install the required packages if they are not already available:

   ```bash
   pip install torch torchvision opencv-python pandas numpy pillow matplotlib scikit-learn scipy tqdm
   ```

4. Make the dataset available in your environment.
5. Update the dataset paths in the notebook. If using Google Colab, mount Google Drive as shown in the notebook.
6. Run the notebook cells in order, checking that the data paths, labels, model setup, and training/evaluation steps are configured correctly for your environment.

A CUDA-enabled GPU is recommended for faster deep-learning experiments. The notebook includes automatic device selection in its main model setup.

## Evaluation

The notebook includes the following evaluation measures:

| Task | Metrics included |
|---|---|
| Gender classification | Accuracy, F1-score, ROC-AUC / ROC curve |
| Diabetes classification | Accuracy, F1-score, ROC-AUC / ROC curve |
| Age estimation | RMSE, MAE in years, Pearson correlation |

**Evaluation caveat:** The current notebook evaluates predictions by iterating over `train_loader`, which is built from the full dataset with shuffling. This is not an independent test evaluation. Before reporting performance as a project result, create train/validation/test splits (ideally patient-independent), train only on the training split, select settings using validation data, and report final metrics on held-out test data. Also verify the training loop and loss functions are present and executed in the notebook.

## Repository Structure

```text
multitask-retinal-fundus-analysis/
├── README.md
└── with_attention_blocks.ipynb
```

Keep the downloaded dataset and trained model weights out of Git unless there is a clear reason and you have confirmed their redistribution terms. The notebook saves a model state dictionary as `multitask_fundus_model.pth` when that cell is run; large generated files can be stored separately.

## Future Improvements

- Add patient-independent train, validation, and test splits.
- Add reproducible random seeds and configuration settings.
- Track each task's loss separately and tune loss weights.
- Handle class imbalance and missing labels explicitly.
- Compare against baseline models and report held-out metrics.
- Improve Grad-CAM visualizations and validate the target layer.
- Save training history and evaluation results in a reproducible format.

## Disclaimer

This repository is for learning and research purposes only. It is not intended to diagnose, screen for, or treat any medical condition. Model outputs may be inaccurate or biased and must not be used as a substitute for professional medical evaluation.

## Dataset Attribution

The ODIR-5K dataset is provided through Kaggle by Andrew M. Please refer to the dataset page for the original description, terms, and attribution requirements: https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k
