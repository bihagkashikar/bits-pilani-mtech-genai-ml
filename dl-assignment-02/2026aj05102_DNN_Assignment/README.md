# DNN Assignment

This workspace contains the completed assignment scaffolding and implementation notes for the DNN programming assignment.

## Structure
- `problem1/` contains the recurrent-model implementation and validation notebook.
- `problem2/` contains the attention and Transformer implementation notebook.
- `results.csv` stores the recorded experiment metrics.
- `generated_samples.txt` contains 500-character samples for the language model.

## Environment
Use Python 3.12 and install the pinned environment with `uv`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name bits-pilani-dnn-assignment --display-name "Python 3.12 (DNN Assignment)"
```

## Running the notebooks
1. Open the notebook in Jupyter or Colab.
2. Select the **Python 3.12 (DNN Assignment)** kernel.
3. Run the cells in order with “Restart and Run All”.
4. Keep the working directory at the assignment root so relative paths resolve correctly.

## Dataset notes
- Problem 1 uses the UCI HAR dataset and the generated adding problem.
- Problem 2 uses HAR again for the encoder classifier and Tiny Shakespeare for the character model.

## GPU fallback
A free Colab T4 is sufficient. If no GPU is available, use the fallback configuration noted in the FAQ: set the Tiny Shakespeare model to `d = 64` and `2` transformer layers, and record that in the README.
