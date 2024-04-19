# CS4248 Project

**Group 14**

This repository contains code used in our evaluation of the performance of different models in classifying citation intent in academic papers.

Our results and analysis, please refer to the [accompanying report](https://github.com/justinfidelis/CS4248-Project/blob/main/Report.pdf)

## Abstract
In this paper, we explore classifying citation intent in research papers (\textit{background, methods, results}), by applying different architectures and models. Specifically, we focus on three areas: how the models compare on a high level, how specific methods can be used to model task complexity more robustly, and which features best capture semantic information that is most helpful for this task. The RNN variants performs best, but with interesting observations about how they interact with the attention mechanism. In particular, we found a nuanced relationship between model complexity and performance beyond the bias-variance tradeoff, while also observing some subtleties about certain features, such as the differential effects of word and word order on predictions. Ultimately, we hope to demonstrate that interesting insights can be derived on these comparatively simple models.

## Setup
This project used python 3.10.
Required packages are specified in the `src/requirements.in`

Use pip to install required dependencies:

```pip install -r requirements.in```

## Models
Implementation of different models can be found in `src/models`

Implemented Models:
* Multinomial Logistic Regression
* Multilayer Perceptron
* Recurrent Neural Network
* Long Short-Term Memory

## Experiments
Experiment code can be found in their associated folder in `src`

1. Comparing model performance
   1. MLP vs LogReg
   2. RNN vs LSTM
2. Model augmentations
   1. Attention layers
   2. Additional hidden layers
3. Feature information
   1. Static features
   2. Word order
