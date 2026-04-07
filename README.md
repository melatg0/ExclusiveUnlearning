# Exclusive Unlearning
This code contains the implementation for Exclusive Unlearning.
It also provides instructions for running experiments in the medical and mathematics settings.

## Environment Setup
The following steps show how to set up the environment using `uv`.
```
cd colm2026
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Downloading the Training Data
For the setting where medical knowledge is retained, download the MedInstruct-52k dataset.
For the setting where mathematical knowledge is retained, download the MetaMathQA dataset.

They can be downloaded from the following sources:
- MedInstruct-52k: https://github.com/XZhang97666/AlpaCare
- MetaMathQA: https://github.com/meta-math/MetaMath

## Sampling Forgetting Texts from the Model Itself
An important feature of Exclusive Unlearning is that, in order to generalize forgetting to unseen harmful inputs, it makes the generation probabilities of texts sampled from the model itself uniform.

Sampling is performed with `src/forget_data_sampling/sampling.py`.

## Running Training
The implementation of Exclusive Unlearning is in `src/exclusive_unlearning/run_exclusive_unlearning.py`.
You can run it as follows for the medical and mathematics retention settings:
- Medical
```
bash src/exclusive_unlearning/run_exclusive_unlearning_alpacare.sh
```
- Mathematics
```
bash src/exclusive_unlearning/run_exclusive_unlearning_metamath.sh
```