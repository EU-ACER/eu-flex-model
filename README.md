<!--
Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System -->

# EU-Flex: An open-source model for assessing electricity flexibility needs in Europe

> [!IMPORTANT]
> The EU-Flex model is still under development. The version of the model published in this repository is preliminary and it is not the version to be used by ACER to run the EU-wide Flexibility Needs Assessment. 

This repository introduces the EU-Flex model. EU-Flex is an open-source electricity-market modelling tool developed by the European Union Agency for the Cooperation of Energy Regulators (ACER) to carry out the EU-wide Flexibility Needs Assessment (FNA), under Article 19e of the [Electricity Market Design][regulation] framework.

The EU-Flex model is based on [Python for Power System Analysis][pypsa] (PyPSA), an open-source toolbox for simulating and optimising modern power and energy systems. Choosing an open-source solution will enhance transparency in energy modelling, encourage dialogue and collaboration among stakeholders, and support robust, evidence-based policymaking. 

Making the preliminary model and its underlying approach publicly available will help stakeholders understand the underlying methodology, provide feedback and contribute to its further development.

Through the development of the EU-Flex model, ACER has created new modelling features that could be considered for upstream integration into PyPSA and/or [PyPSA-Eur][pypsa-eur], the open-source model of the European electricity system. ACER intends to make these developments available to the wider modelling community, fostering further collaboration and supporting the continued development of open-source energy modelling tools.

The model has been developed with the support of [Open Energy Transition][oet] (OET) and the European Commission Joint Research Centre.

## About the EU-Flex model

EU-Flex is a model that simulates the European power market. It is designed to assess flexibility needs across Europe in a transparent and replicable way.

The model considers two market timeframes (day-ahead and intraday-balancing), allowing for a more granular representation of power system operations and a more accurate assessment of the revenue streams available to assets in the market. Based on these revenues, the model simulates how the European resource mix could evolve over time. The model then compute:

- electricity-system flexibility needs indicators in line with the methodology specified in [ACER Decision No 05/2025][fna-decision] on the EU-wide Flexibility Needs Assessment.
- contributions to flexibility needs per technology (including cross-border interconnectors); and 
- other market indicators (e.g. prices).

For a detailed description of the model architecture, equations, assumptions and indicators, see: [Modelling Specification](doc/modelling_specification.pdf).

## Disclaimer

The EU-Flex model is still under development. The version of the model published in this repository is preliminary and it is not the version to be used by ACER to run the EU-wide Flexibility Needs Assessment. The purpose of this preliminary release is to allow users to familiarise with the model, review its key features, and provide constructive feedback on it. 

To ensure the preliminary version can run easily, ACER provides a test dataset that displays the model’s key components. Please note that this test dataset has been developed for illustrative purposes only. It is not intended to represent actual or forecast market conditions, and results generated using it should not be interpreted as conclusions concerning flexibility needs, market developments or investment requirements in the European electricity system.

The model remains under development and may be modified following further testing and stakeholder feedback. Neither the preliminary model nor the results generated using the illustrative dataset constitute ACER's final assessment, findings or recommendations under Article 19e(7) of Regulation (EU) 2019/943.

While reasonable care has been taken in preparing the material, ACER shall not be liable for any loss or damage arising from their use, except as provided by applicable law.


## Repository structure

The model runs as a [Snakemake][snakemake] workflow. This repository contains:

- the model code and its configuration (`src/`, `config/`);
- the teaser model, the test case of the ACER paper "Preparing for the European flexibility needs assessment" (ACER, 2026), in `data/teaser/`;
- a notebook for analysing the results of the teaser model (`notebooks/`);
- a [User Manual](doc/user_manual.md) ([PDF](doc/user_manual.pdf)), which explains the model's options, outputs and troubleshooting in detail;
- the [Modelling Specification](doc/modelling_specification.pdf), the mathematical formulation of the model.

## Installation

You need an internet connection and a few gigabytes of free disk space for the installation. The model runs on:

| Platform | Tested |
|:---|:---|
| Windows 10 and 11, x86-64 (Intel or AMD) | yes |
| Linux, x86-64 | yes |
| macOS, Apple silicon (M1 and later) | yes |
| macOS, Intel | no |

Follow the steps for your system. Copy each command, paste it into the terminal and press Enter. Wait for it to finish before you paste the next one.

### Windows

1. **Install Git.** Download [Git for Windows][git-for-windows] and run the installer, accepting the default options.
2. **Install conda.** Download [Miniforge for Windows][miniforge-windows] and run the installer, accepting the default options. If you already have Miniconda or Anaconda, you can skip this step.
3. **Open the Miniforge Prompt** from the Start menu (with Miniconda or Anaconda: the *Anaconda Prompt*). Run all the commands below in this window.
4. **Download the model** to `C:\eu-flex-model` (or a different folder of your choosing):

   ```bat
   cd C:\
   git clone https://github.com/EU-ACER/eu-flex-model.git eu-flex-model
   cd eu-flex-model
   ```

5. **Create the environment.** This downloads the packages the model needs and can take several minutes:

   ```bat
   conda env create -f envs/win-64.lock.yaml
   conda activate eu-flex
   pip install -e .
   ```

### Linux

1. **Install Git** with your package manager, for example on Ubuntu or Debian:

   ```bash
   sudo apt install git
   ```

2. **Install conda.** If you already have Miniforge, Miniconda or Anaconda, skip this step. Otherwise run these two commands, answer `yes` to each question, then close the terminal and open a new one:

   ```bash
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```

3. **Download the model:**

   ```bash
   git clone https://github.com/EU-ACER/eu-flex-model.git eu-flex-model
   cd eu-flex-model
   ```

4. **Create the environment.** This downloads the packages the model needs and can take several minutes:

   ```bash
   conda env create -f envs/linux-64.lock.yaml
   conda activate eu-flex
   pip install -e .
   ```

### macOS

1. **Install Git.** Open the *Terminal* app and run `git --version`. If Git is not installed, macOS offers to install it: accept, and wait for it to finish.
2. **Install conda.** If you already have Miniforge, Miniconda or Anaconda, skip this step. Otherwise run these two commands, answer `yes` to each question, then close the terminal and open a new one:

   ```bash
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```

3. **Download the model:**

   ```bash
   git clone https://github.com/EU-ACER/eu-flex-model.git eu-flex-model
   cd eu-flex-model
   ```

4. **Create the environment.** This downloads the packages the model needs and can take several minutes. On a Mac with Apple silicon (M1 and later):

   ```bash
   conda env create -f envs/osx-arm64.lock.yaml
   conda activate eu-flex
   pip install -e .
   ```

   On a Mac with an Intel processor, use `envs/osx-64.lock.yaml` in the first command instead.

### Check the installation

On every system, still in the `eu-flex-model` folder, run:

```bash
snakemake -n -c1 --configfile config/teaser.yaml --directory runs/teaser
```

This plans a run of the test case / teaser model without running it. It prints the steps it would run, followed by a table counting them per rule. If it gets that far without an error, the installation works.

## Running the teaser model

Every time you open a new terminal (on Windows: Miniforge Prompt) to use the model, first go to the `eu-flex-model` folder and activate the environment:

```bat
cd C:\eu-flex-model
conda activate eu-flex
```

On Linux and macOS, `cd` to wherever you downloaded the model instead, for example `cd ~/eu-flex-model`, followed by `conda activate eu-flex`.

In this pre-release, we show three ways of running the model. The commands are the same on every system:

1. **Flexibility indicators** on the network as supplied:

   ```bash
   snakemake -c all --configfile config/teaser.yaml --directory runs/teaser
   ```

2. **Economic viability assessment**, which first adjusts the capacities over up to 30 iterations:

   ```bash
   snakemake -c all --configfile config/teaser-eva.yaml --directory runs/teaser-eva
   ```

3. **Dummy storage**, which sizes the storage needed to meet a range of curtailment targets, once with storage shared across countries and once without:

   ```bash
   snakemake -c all --configfile config/teaser-ds-cross-border.yaml --directory runs/teaser-ds-cross-border
   snakemake -c all --configfile config/teaser-ds-no-cross-border.yaml --directory runs/teaser-ds-no-cross-border
   ```

How long a run takes depends on your computer. As a rough guide, the first run takes in the order of ten minutes. The second does a similar amount of work in each of its iterations, so it can take several hours. Each dummy storage run solves 63 optimisations of a few minutes each, about three and a half hours of computing in total, so with a few running in parallel it takes in the order of an hour. Each optimisation needs up to 2 GB of memory, so on a machine with little memory use a lower `-c`, such as `-c 4`. A dummy storage run also writes about 9 GB of model files.

The results of the respective runs can be found in:

- `runs/teaser/results/assessment/model_outputs.nc` for the first run;
- `runs/teaser-eva/results/assessment/model_outputs.nc` for the second;
- `runs/teaser-ds-cross-border/results/assessment/model_outputs.nc` and `runs/teaser-ds-no-cross-border/results/assessment/model_outputs.nc` for the dummy storage runs.

Each file holds all the indicators of its run, and for the dummy storage runs also the results of the dummy storage optimisation. The [User Manual](doc/user_manual.md) describes what is in it and how to open it in Python, and how to change the model's settings. If a command fails, see its Troubleshooting section.

With the notebook `notebooks/teaser_case_analysis.ipynb` you can reproduce Figures 9 to 18 of the ACER Paper from the results of the first run, and make plots of your own. Section 2.8 of the [User Manual](doc/user_manual.md) explains how to open and run it.

## Feedback

The model published in this repository has been released as part of ACER's public consultation on the preliminary model for assessing electricity flexibility needs in the EU. As such, stakeholders are invited to provide their feedback before **17 December 2026** when the public consultation will officially close.

This preliminary release will give stakeholders an opportunity to review the proposed modelling approach and contribute to the development of the EU-Flex model ahead of the 2027 assessment, and beyond. Stakeholders are invited to provide their constructive feedback on the preliminary version of EU-Flex in this repository as described below.

Please note that only severe bug or error reports preventing users from installing and running the model will be addressed as soon as feasible. All other feedback will be collected and reviewed by ACER. A summary of the feedback collected and how ACER has addressed it will be provided in the EU-wide FNA report, planned to be published in July 2027.

To give feedback on the model, its code or its documentation:

1. Go to the [Issues](https://github.com/EU-ACER/eu-flex-model/issues) page of this repository.
2. Click **New issue**.
3. Choose a template: **Bug Report** for errors or unexpected behaviour, **Feature Request** for suggestions, **Question** for help with using the model or **Documentation Feedback** for feedback on the README and User Manual.
4. Fill in the template and submit it.

For feedback on the methodology (the Modelling Specification) or on the ACER paper, please use the ACER survey, which is communicated separately.

## Licence and data sources

The code and model configuration are released under the MIT licence, the documentation and the teaser model under CC-BY-4.0, and repository settings, environment files and other input data under CC0-1.0. Some of the code is adapted from [PyPSA-Eur][pypsa-eur]. The licence texts are in the `LICENSES/` directory, and `REUSE.toml` and the file headers record which licence applies to which file, following the [REUSE][reuse] specification.

[regulation]: https://eur-lex.europa.eu/eli/reg/2024/1747
[pypsa]: https://pypsa.org/
[pypsa-eur]: https://pypsa-eur.readthedocs.io/en/latest/
[oet]: https://openenergytransition.org/
[snakemake]: https://snakemake.readthedocs.io/en/stable/
[fna-decision]: https://www.acer.europa.eu/sites/default/files/documents/Individual%20Decisions/ACER-Decision-05-2025-Flexibility-Needs-Assessment-Methodology.pdf
[git-for-windows]: https://git-scm.com/install/windows
[miniforge-windows]: https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe
[reuse]: https://reuse.software/
