# Evaluation Metrics

This document provides the mathematical definitions of the performance metrics used in the study **“Automatic Arch Recognition in Architectural Heritage Images Using Transfer Learning and Architecture-Aware Data Augmentation.”** These definitions complement Section 3.7 of the manuscript.

## Binary classification setting

The classification task contains two classes:

- **Has Arch** — positive class
- **No Arch** — negative class

The following notation is used:

- **TP (True Positive):** Has Arch image correctly classified as Has Arch
- **TN (True Negative):** No Arch image correctly classified as No Arch
- **FP (False Positive):** No Arch image incorrectly classified as Has Arch
- **FN (False Negative):** Has Arch image incorrectly classified as No Arch

## Accuracy

Accuracy represents the proportion of correctly classified images among all evaluated images:

$$
\mathrm{Accuracy} =
\frac{TP + TN}
{TP + TN + FP + FN}
$$

## Precision

Precision represents the proportion of predicted positive samples that are correctly classified:

$$
\mathrm{Precision} =
\frac{TP}
{TP + FP}
$$

## Recall

Recall represents the proportion of positive samples that are correctly identified:

$$
\mathrm{Recall} =
\frac{TP}
{TP + FN}
$$

## F1-score

The F1-score is the harmonic mean of precision and recall:

$$
\mathrm{F1} =
2 \times
\frac{\mathrm{Precision} \times \mathrm{Recall}}
{\mathrm{Precision} + \mathrm{Recall}}
$$

## Class-specific metrics

Precision, recall, and F1-score were calculated separately for both classes. When the **No Arch** class was evaluated as the target class, the class labels were interpreted accordingly.

## Macro-averaged F1-score

Macro F1 assigns equal importance to both classes and is calculated as the arithmetic mean of the two class-specific F1-scores:

$$
\mathrm{Macro\ F1} =
\frac{\mathrm{F1}_{\mathrm{Has\ Arch}} +
\mathrm{F1}_{\mathrm{No\ Arch}}}{2}
$$

## Confusion matrix

The confusion matrix was used to summarize the distribution of true-positive, true-negative, false-positive, and false-negative predictions. It also supports the qualitative analysis of classification errors reported in the manuscript.

## Reproducibility note

These equations are standard definitions of classification metrics and are provided here as supplementary methodological documentation for reproducibility and transparency. The implementation used the same class definitions and held-out test predictions reported in the manuscript.
