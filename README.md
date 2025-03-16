# FreqMoE: Enhancing Time Series Forecasting through Frequency Decomposition Mixture of Experts (AISTATS 2025 poster)
The official implementation of "FreqMoE: Enhancing Time Series Forecasting through Frequency Decomposition Mixture of Experts".

[[Arxiv]](https://arxiv.org/abs/2501.15125)

## TLDR
In this project we propose **FreqMoE**, a frequency-based Mixture of Experts model for long-term time series forecasting. Unlike existing methods, FreqMoE dynamically decomposes time series into frequency bands, with specialized experts processing each band. A gating mechanism adjusts expert contributions, and a prediction module refines forecasts via residual connections. Experiments show FreqMoE achieves SOTA performance across eight datasets while keeping parameters under **50k**, ensuring high efficiency.

## Datasets Preparation
You can access all nine benchmark datasets from the [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) link provided in Autoformer. These datasets are well pre-processed and readily usable. Please download the datasets and put them in the ```./dataset``` folder. Each dataset is an ```.csv``` file.

## Environment requirements
```python
pip install -r requirements.txt
```
Please refer to the ```requirements.txt``` file for the required packages.
