# NoisyCLUE

Implementation of the experiments for the NoisyCLUE dataset.

## Models

- MacBERT
- mT5
- ByT5

## Dataset

- AFQMC
- CLUENER
- CMRC2018
- CSpider
- newstest

## Results

### AFQMC balanced

model                           | acc clean | acc noisy 1 | acc noisy 2 | acc noisy 3 | avg | worst group | macro f1 clean | f1 0 clean | f1 1 clean | f1 0 noisy 1 | f1 1 noisy 1 | f1 0 noisy 2 | f1 1 noisy 2 | f1 0 noisy 3 | f1 1 noisy 3
----------------------------    | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- 
byt5-base_lr2e-4                | 45.20 | 44.95 | 45.69 | 45.16 | 45.27 | 29.36 | 45.19 | 44.29 | 46.09 | 44.12 | 45.75 | 45.64 | 45.74 | 45.09 | 45.22
byt5-small_lr2e-4               | 36.54 | 40.08 | 38.35 | 38.65 | 39.03 | 28.66 | 33.47 | 19.18 | 47.76 | 30.71 | 47.22 | 25.19 | 47.57 | 26.61 | 47.29
chinese-macbert-base_lr2e-5     | 70.46 | 67.52 | 70.11 | 69.16 | 68.93 | 47.73 | 68.44 | 76.42 | 60.47 | 76.61 | 46.85 | 79.41 | 45.48 | 78.12 | 47.78
chinese-roberta-wwm-ext_lr5e-5  | 68.88 | 68.28 | 69.67 | 68.74 | 68.90 | 48.33 | 67.22 | 74.60 | 59.85 | 77.60 | 45.70 | 79.19 | 44.13 | 78.19 | 44.87
mt5-base_lr2e-4                 | 61.19 | 63.53 | 66.15 | 63.92 | 64.54 | 42.52 | 60.70 | 65.08 | 56.32 | 72.08 | 47.43 | 74.80 | 48.47 | 72.36 | 48.08
mt5-base_lr3e-4                 | 67.96 | 67.05 | 68.33 | 68.12 | 67.83 | 63.16 | 45.26 | 80.51 | 10.02 | 79.82 | 10.23 | 80.89 | 7.45 | 80.67 | 8.99
mt5-small_lr2e-4                | 68.05 | 68.03 | 68.51 | 68.00 | 68.18 | 62.40 | 50.26 | 80.01 | 20.52 | 80.34 | 14.39 | 80.78 | 12.94 | 80.29 | 15.12



