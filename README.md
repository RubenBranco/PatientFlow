# PatientFlow: Learning to Generate Mixed-Type Longitudinal Clinical Data with Flow Matching

<p align="center">
  <img src="docs/assets/PatientFlow.png" alt="PatientFlow Architecture" width="700"/>
</p>

<p align="center">
  <strong>Ruben Branco¹, Marta Gromicho², Mamede de Carvalho², Piero Fariselli³, Sara C. Madeira²</strong>
</p>
<p align="center">
  <sup>1</sup>LASIGE, Faculdade de Ciências, Universidade de Lisboa, Campo Grande, Lisboa, 1749-016, Portugal<br>
  <sup>2</sup>Faculdade de Medicina, Universidade de Lisboa, Av. Prof. Egas Moniz, Lisboa, 1649-028, Portugal<br>
  <sup>3</sup>Department of Medical Sciences, University of Torino, Corso Dogliotti 14, Turin, 10126, Italy
</p>
<p align="center">
  📧 rmbranco [at] fc.ul.pt
</p>

---

## Abstract

Synthetic longitudinal clinical data can help unlock large-scale deep learning models to tackle complex diseases. However, learning to generate realistic samples faces dual challenges: modeling the inherently complex structure of longitudinal mixed-type data and protecting patient privacy.

We introduce **PatientFlow**, a generative modeling method combining Variational Autoencoders for data representation with Flow Matching for sample generation. We extensively evaluated the model on a longitudinal cohort of patients with Amyotrophic Lateral Sclerosis (*N* = 1,560) using both qualitative and quantitative methods.

The model demonstrated an ability to generate realistic samples, which was further validated by expert clinicians. Prognosis models trained on our synthetic data across five clinically relevant endpoints matched and sometimes exceeded the performance of models trained on real data.

Our results demonstrate that PatientFlow can effectively model longitudinal clinical data with high fidelity, opening promising avenues for sharing and augmenting datasets for deep learning applications in healthcare.

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

### Additional Dependencies

Our extension of the Multi-Sequence Aggregate Similarity, used for quantitative analysis, can be found and installed [here](https://github.com/RubenBranco/msas-pytorch).

```bash
# Install eMSAS for advanced similarity metrics
pip install git+https://github.com/RubenBranco/msas-pytorch.git
```

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

1. Extend the `LightningDataModule` class as shown in `BrainteaserDataModule`. Ensure it has the necessary properties for the autoencoder to work with (e.g. `.features` property of type `FeatureList`)
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
   - Quantitative evaluation of synthetic data quality with [eMSAS](https://github.com/RubenBranco/msas-pytorch) and Prognostic Metrics
   - Parallelized computation for efficient evaluation across multiple synthetic datasets

3. **privacy.ipynb**
   - Privacy analysis of the generated synthetic data

4. **semantic_analysis.ipynb**
   - Analysis of semantic preservation in the synthetic data
   - Clinical plausibility assessment using domain-specific rules

5. **statistical_tests.ipynb**
   - Statistical comparison between original and synthetic datasets
   - Comprehensive hypothesis testing including KS tests, t-tests, chi-square tests, and Fisher's exact tests
   - Automated LaTeX table generation for statistical results

6. **clinical_analysis_sample.ipynb**
   - Generation of balanced samples (real vs. synthetic patients) for clinical evaluation
   - Excel workbook creation with structured evaluation forms for clinical experts

7. **clinical_analysis.ipynb**
   - Analysis of clinical expert evaluation results
   - Confusion matrix generation and statistical analysis of expert discrimination ability
   - Confidence level analysis and reasoning categorization

## Citation

```bibtex
@article{BRANCO2026103392,
  title    = {PatientFlow: Learning to generate mixed-type longitudinal clinical data with flow matching},
  journal  = {Artificial Intelligence in Medicine},
  volume   = {176},
  pages    = {103392},
  year     = {2026},
  issn     = {0933-3657},
  doi      = {https://doi.org/10.1016/j.artmed.2026.103392},
  url      = {https://www.sciencedirect.com/science/article/pii/S0933365726000448},
  author   = {Ruben Branco and Marta Gromicho and Mamede {de Carvalho} and Piero Fariselli and Sara {C. Madeira}},
  keywords = {Deep learning, Generative modeling, Flow matching, Longitudinal clinical data, Prognosis},
  abstract = {Synthetic longitudinal clinical data, with static and temporal mixed-type components, can help unlock large-scale deep learning models to tackle complex diseases. However, learning to generate realistic patients faces dual challenges: modeling the inherently complex structure of longitudinal data and protecting patient privacy. We introduce PatientFlow, a generative modeling method combining Variational Autoencoders for data representation with Flow Matching for patient generation. We extensively evaluated the generative model on a longitudinal cohort of patients with Amyotrophic Lateral Sclerosis (N = 1560) using both qualitative and quantitative methods. The ability of the method to generate realistic patient data, further validated by expert clinicians, shows its potential application to other diseases. Prognostic models trained on synthetic data across five clinically relevant endpoints matched and sometimes outperformed the models trained on real data. Our results demonstrate that PatientFlow can effectively model longitudinal clinical data with high fidelity, opening promising avenues for sharing and augmenting datasets for deep learning applications in healthcare without compromising privacy.}
}
```