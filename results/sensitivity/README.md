# ResNet50 exact-overlap sensitivity analysis

This analysis evaluates the saved ResNet50 checkpoint without retraining after excluding 21 test images with byte-identical counterparts in the training or validation subsets.

The original test set contains 1,496 images. The sensitivity subset contains 1,475 images after excluding 17 test images overlapping with training and four overlapping with validation.

## Sample membership

The corresponding files are in the repository's data directory:

- ../../data/sensitivity_test_ids.csv: identifiers of the 1,475 retained test samples.
- ../../data/excluded_21_exact_overlaps.csv: identifiers of the 21 excluded test samples and recorded audit links.

Sample identifiers can be joined to ../../data/fixed_test_split_public.csv for filenames and class labels: 0 = No Arch; 1 = Has Arch.

The original fixed test partition remains unchanged. This post-hoc sensitivity analysis supplements the original evaluation and does not constitute an independent external test.
