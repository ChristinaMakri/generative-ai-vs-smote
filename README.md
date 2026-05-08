# Generative AI vs SMOTE — Imbalanced Data Benchmark

> Can generative AI outperform SMOTE on imbalanced medical data? A systematic comparison of synthetic oversampling techniques.

## Research Question

When dealing with imbalanced tabular data, classical resampling methods like SMOTE have been the go-to solution. This project investigates whether modern generative models (CTGAN, TVAE, custom GAN) can produce higher-quality synthetic minority samples — and whether that translates to better downstream classifier performance.

**Dataset:** [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) — ~5,100 records, ~5% stroke cases (~20:1 imbalance ratio)

## Approach

We evaluate each method on two levels:

1. **Synthetic data quality** — Do the generated samples resemble real stroke cases? (distribution similarity, feature correlations)
2. **Downstream classifier performance** — Does augmentation with synthetic data improve F1, PR-AUC, and MCC?

## Methods Compared

| Method | Type | Description |
|---|---|---|
| No resampling | Baseline | Raw imbalanced data |
| Class weights | Baseline | Penalize majority class |
| SMOTE | Classical | Interpolation between minority samples |
| CTGAN | Generative | GAN-based tabular data synthesizer |
| TVAE | Generative | VAE-based tabular data synthesizer |
| Custom GAN | Generative | PyTorch GAN trained on minority class |

## Project Structure

```
├── data/                        # Dataset (not tracked in git)
├── notebooks/
│   ├── 1_eda.ipynb              # Exploratory Data Analysis
│   ├── 2_baselines.ipynb        # SMOTE & classical baselines
│   ├── 3_generative/
│   │   ├── ctgan.ipynb          # CTGAN & TVAE (SDV library)
│   │   └── custom_gan.ipynb     # Custom PyTorch GAN
│   ├── 4_evaluation.ipynb       # Full benchmark comparison
│   └── 5_synthetic_quality.ipynb # Synthetic data quality analysis
└── src/                         # Reusable modules
```

## Setup

```bash
git clone https://github.com/ChristinaMakri/generative-ai-vs-smote.git
cd generative-ai-vs-smote
pip install -r requirements.txt
```

Download the dataset from Kaggle and place it at `data/healthcare-dataset-stroke-data.csv`:

```bash
kaggle datasets download -d fedesoriano/stroke-prediction-dataset -p data/ --unzip
```

## Evaluation Metrics

Accuracy is misleading on imbalanced data. We use:
- **PR-AUC** — Precision-Recall Area Under Curve
- **F1 Score** — Harmonic mean of precision and recall
- **MCC** — Matthews Correlation Coefficient

## Stack

- Python, PyTorch
- scikit-learn, imbalanced-learn
- SDV (CTGAN, TVAE)
- pandas, matplotlib, seaborn
