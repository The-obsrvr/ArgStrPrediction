# Argument Structure Prediction in Conversations: A Systematic Evaluation

This repository contains documentation on a project that forms part of a broader PhD research effort titled "Identifying the Stance of Argumentative Opinions in Political Discourse", conducted under the HYBRIDS Project within the Horizon Europe framework.

The primary contributor and point of contact for this repository is Siddharth Bhargava (sbhargava@fbk.eu).
---

## Overview

The main contributions of this work are as follows:

- Introduces a systematic adaptation of IAT-based dialogical corpora into simplified bipolar argument structures, facilitating consistent benchmarking and computational modeling;
- Develops an end-to-end pipeline for Argument Structure Prediction across multiple task architectures (single-step and multi-step) and modeling paradigms (fine-tuning and prompt-based approaches);
- Proposes a comprehensive evaluation framework for analyzing performance, generalization, schema compliance, and computational efficiency of argument structures across modeling configurations under shared schema.

---

## Data

We use AIFdb, a large repository of dialogical argument mining corpora annotated under Inference Anchoring Theory (IAT) by different research teams, and represented in the Argument Interchange Format  (AIF).

For detailed documentation on how the IAT annotations are processed into Bipolar Argument Structures refer to our Data Pipeline repository here: [IAT-BAS-Data-Pipeline](https://github.com/The-obsrvr/IAT-BAS-Data-Pipeline). 

###### Add Corpus Stats as image

---

## Methodology

###### Add modeling configs as image

### Single-step Fine-tuning Deep Learning Models

### Multi-step Fine-tuning Deep Learning Models

### Single-step Prompting Large Language Models

### Multi-step Prompting Large Language Models

---

## Implementation

### Environment and Project Initialization

Build the Docker Container using the following command
```shell
$ docker build --rm -t YOUR_CONTAINER_NAME . 
```
Thia builds a Docker container containing the project environment. All experiments have been executed on a Linux server with a 48GB NVIDIA Ampere A40 GPU, CUDA version: 12.4 and Python version: 3.11.

### Execution Command

The command is run in the docker environment as follows:
```shell
$ docker run   --gpus='"device=DEVICE_NUMBER"' --runtime=nvidia --rm -ti --shm-size=32gb -v $PWD:/app YOUR_CONTAINER_NAME ./exe.sh 
```

The ```exe.sh``` for each configuration is defined as follows:

#### Single-step Fine-tuning Deep Learning Models

#### Multi-step Fine-tuning Deep Learning Models

#### Single-step Prompting Large Language Models

#### Multi-step Prompting Large Language Models


---

## Evaluation 

### Task Performance

#### Generalization

### Computational Efficiency

#### Schema Validation 

### Error Analysis 

#### Unit Segmentation and Alignment

#### Relation Prediction

#### Relation Type Classification

---

## Acknowledgements

This research work has received funding from the European Union's Horizon Europe research and innovation programme under the Marie Skłodowska-Curie Grant Agreement No. 101073351. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Research Executive Agency (REA). Neither the European Union nor the granting authority can be held responsible for them.

---

## Citation 

tbd
