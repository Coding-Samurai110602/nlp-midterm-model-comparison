# Results

All models: character-level Tiny Shakespeare, context = 128, batch = 64, AdamW, trained on a MacBook Air (MPS).

## Main comparison (3000 steps each)

| Model | Params | Train Loss | Val Loss | Val Perplexity | Training Time (s) | Generated Text Quality | Notes |
|---|---|---|---|---|---|---|---|
| Bigram | 4,225 | 2.455 | 2.483 | 11.97 | 14.4 | Low | Plausible letter pairs but almost no real words |
| LSTM | 946,625 | 1.151 | **1.513** | **4.54** | 156.9 | Medium–High | Best val loss; validation loss rose after step 2250 (overfitting) |
| Transformer | 824,897 | 1.341 | 1.560 | 4.76 | 376.5 | Medium–High | Still improving at step 3000; smaller train/val gap (0.22 vs 0.36) |

## Experiment: longer Transformer training

| Model | Steps | Final Val Loss | Best Val Loss (step) | Training Time (s) |
|---|---|---|---|---|
| LSTM | 3000 | 1.513 | 1.4995 (2250) | 156.9 |
| Transformer | 3000 | 1.560 | 1.560 (3000) | 376.5 |
| Transformer | 6000 | 1.514 | 1.513 (5250) | 875.1 |
