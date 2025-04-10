# PatientFlow: A Generative Framework for Longitudinal Patient Data

<p align="center">
  <img src="docs/assets/PatientFlow.png" alt="PatientFlow Architecture" width="700"/>
</p>

<p align="center">
  <strong>Author Name¹², Author Name², Author Name³</strong>
</p>
<p align="center">
  <sup>1</sup>Department of Computer Science, University Name<br>
  <sup>2</sup>Medical Research Institute, University Name<br>
  <sup>3</sup>Department of Medical Informatics, University Name
</p>
<p align="center">
  <a href="mailto:corresponding.author@email.com">corresponding.author@email.com</a>
</p>

---

Abstract: <em>abstract here</em>

*<abstract here>*

## Overview

PatientFlow is a generative framework for modeling longitudinal patient data using flow matching and variational autoencoder techniques. The framework is designed to generate realistic synthetic patient trajectories while preserving the statistical properties of the original data. This approach enables researchers to share synthetic datasets, and augment existing ones, that still maintain the utility of real patient data without compromising privacy.

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/RubenBranco/PatientFlow.git
cd PatientFlow

# Install the basic package
pip install -e .
```

### Installation with Experiment Dependencies

To run the notebooks and experiments, additional dependencies are required:

```bash
# Install with extra dependencies for experiments
pip install -e ".[experiments]"
```

Our extension of the Multi-Sequence Aggregate Similarity, used for quantitative analysis, can be found and installed [here](https://github.com/RubenBranco/msas-pytorch).

## Data Format and Custom DataModules

PatientFlow is designed to be flexible regarding data input formats. While the provided implementation uses CSV files, the framework can be adapted to work with various data sources by creating custom DataModules following the structure in `patientflow/data.py`.

### Input Data Structure for Models

The PatientFlow VAE model expect data to be structured as follows:

1. **Static Data**: A tensor of shape `(batch_size, static_features)` where each row represents a patient and each column represents a static feature
2. **Temporal Data**: A tensor of shape `(batch_size, sequence_length, temporal_features)` where:
   - Each patient has a sequence of observations
   - `sequence_length` is the maximum number of timepoints (padded if necessary)
   - `temporal_features` are measurements that change over time

Creating a custom DataModule that produces these tensor formats allows PatientFlow to work with any type of longitudinal data.

### Custom DataModule Implementation

To implement a custom DataModule:

1. Extend the `LightningDataModule` class as shown in `BrainteaserDataModule`
2. Implement the required methods for data loading, processing, batching, and other necessary operations
3. Ensure your data is formatted into the expected static and temporal tensors

### Experimental Dataset

In our paper, we used a dataset of Amyotrophic Lateral Sclerosis (ALS) patients, collected at the Lisbon ALS clinic (Centro Hospitalar Lisboa Norte), consisting of a longitudinal cohort of 1560 patients regularly followed at the clinic. It is structured as a CSV file with each row representing an observation for a patient at a specific timepoint. The static columns (Gender, Age, NIV, ...) all have the same value across the referenced patient, while the temporal ones may change (P1 through P12).

Below is an example of this data format (synthetic example):

| REF | medianDate  | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10 | P11 | P12 | Gender | Age | NIV | Onset | Ethnicity |
|-----|-------------|----|----|----|----|----|----|----|----|----|----|-----|-----|--------|-----|-----|-------|----------|
| 001 | 2022-01-15  | 4  | 3  | 3  | 4  | 3  | 4  | 2  | 3  | 4  | 3  | 3   | 4   | M      | 65  | 0   | Limb  | White    |
| 001 | 2022-04-20  | 3  | 3  | 3  | 3  | 3  | 3  | 2  | 2  | 3  | 3  | 3   | 3   | M      | 65  | 0   | Limb  | White    |
| 001 | 2022-07-10  | 2  | 2  | 2  | 3  | 2  | 2  | 1  | 1  | 2  | 2  | 2   | 2   | M      | 65  | 0   | Limb  | White    |
| 002 | 2023-02-05  | 4  | 4  | 4  | 4  | 4  | 4  | 4  | 4  | 4  | 4  | 4   | 4   | F      | 58  | 0   | Bulbar| Asian    |
| 002 | 2023-05-15  | 3  | 4  | 3  | 4  | 3  | 3  | 3  | 3  | 3  | 3  | 3   | 3   | F      | 58  | 0   | Bulbar| Asian    |

Where:

- `REF`: Patient identifier
- `medianDate`: Date of the observation
- `P1`-`P12`: Temporal clinical evaluations (e.g., ALS Functional Scores)
- Static features: Features that remain constant for each patient (e.g., Gender, Ethnicity)

## Project Structure

The repository is organized as follows:

```
PatientFlow/
├── docs/
│   └── assets/               # Promotional website
├── patientflow/              # Core package
│   ├── models/               # Model implementations
│   │   ├── ae.py             # Autoencoder models
│   │   └── vector_fields.py  # Vector field implementations
│   ├── data.py               # Data handling utilities
│   └── metrics.py            # Evaluation metrics
├── evaluation_notebooks/     # Notebooks for model evaluation
├── train_notebooks/          # Notebooks for model training
└── setup.py                  # Package installation script
```

## Notebooks

### Training Notebooks

1. **train_vae.ipynb**
   - Training of variational autoencoders for patient data

2. **train_flow_matching.ipynb**
   - Training of static and temporal flow matching networks

### Evaluation Notebooks

1. **distribution_plots.ipynb**
   - Visualization of original vs. synthetic data distributions
   - Feature-level comparisons and distribution plots
2. **metrics.ipynb**
   - Quantitative evaluation of synthetic data quality with [eMSAS](https://github.com/RubenBranco/msas-pytorch) and Prognostic Metrics.
3. **privacy.ipynb**
   - Privacy analysis of the generated synthetic data
4. **semantic_analysis.ipynb**
   - Analysis of semantic preservation in the synthetic data
   - Clinical plausibility assessment
5. **statistical_tests.ipynb**
   - Statistical comparison between original and synthetic datasets
   - Hypothesis testing for distribution similarity

## Citation

Citation coming soon.

<!-- ## License

[Add your license information here]

## Acknowledgments

[Add your acknowledgments here] -->