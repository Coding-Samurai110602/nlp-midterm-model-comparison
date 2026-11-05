# AI Use Summary

**Tool:** Claude (Anthropic)

## What Claude did

- **Planning:** helped me plan the overall setup (character-level Tiny Shakespeare, LSTM as the non-transformer model, PyTorch, a 3000-step budget shared by all models).
- **Model explanations:** explained how the three models work (Bigram, LSTM, and Transformer), which I used to build my understanding and write the explanations in my notebook.
- **Code review and debugging:** reviewed my code, pointed out mistakes, suggested corrections, and helped me solve errors, including terminal commands for project setup and for freeing disk space when training failed.
- **Documentation:** wrote the README.
- **Editing:** reviewed my notes for the "My own understanding" section, corrected one mistake in my reasoning, and helped turn them into full sentences.

## What I did

- Understood the problem statement and decided how to approach it.
- Wrote all of the notebook code: data loading, training/evaluation/generation helpers, the three model classes, the results table, plots, the attention heatmaps, the 6000-step experiment, and the attention-distance measurement.
- Created the project folder and Python environment, installed the dependencies, and downloaded the dataset.
- Ran all training for the three models and the 6000-step experiment on my own MacBook Air. Every number and sample in this project comes from those runs.
- Wrote the notebook's analysis and interpretation of my results.
- Debugged problems along the way, including a folder navigation error during setup and training failing because my disk was full (I found what was using space and cleared it, with Claude's suggestions).
- Uploaded my notebook at each stage for Claude to check, which caught a mistake where I had pasted the LSTM explanation twice and the Transformer explanation was missing.
- Decided which experiments to run and when to move on to each part of the project.

## My own understanding

The bigram learns how often one character follows another, but it only sees the current character, so after "e" it can't tell whether it's in "the", "be" or "Romeo" and can't build real words. Its loss leveled off around 2.48 because that's the best one character of context allows. The LSTM and Transformer both use 128 characters of context, but differently: the LSTM squeezes everything into a fixed hidden state passed forward one step at a time, so older information fades, while the Transformer's attention looks back directly. In my heatmap, letters inside "GREMIO" put nearly all their weight on one earlier letter of the name in a single step. At 3000 steps the LSTM won (1.51 vs 1.56), but its validation loss started rising while training loss kept falling, a sign of overfitting, whereas the Transformer was still improving. With 6000 steps the Transformer reached 1.514, roughly tying the LSTM, but never beat the LSTM's best (1.499) and took 875s vs 130s. On this dataset, the two models end up about even, and the LSTM gets there much faster.

## Verification

All code was written by me and all results were produced by running the notebook myself. Claude's help was limited to planning, explaining the models, reviewing and correcting my code, troubleshooting errors, writing the README, and editing my notes for the understanding section.