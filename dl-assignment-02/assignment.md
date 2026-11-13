# DNN Programming Assignment

## Overview

### How the request is read.

Problem 1 is purely recurrent (vanilla RNN, LSTM and GRU on sequences, no CNN). The Transformer is a separate problem. If you meant a different split, the structure below is easy to change.

### Learning outcomes. A student who finishes can:

• implement RNN, LSTM and GRU cells from their equations and prove each matches the framework's version;
• compare RNN, LSTM and GRU on real sensor sequences and a long-memory task, in accuracy, parameters, training time and gradient behaviour;
• build scaled dot-product attention, multi-head attention, positional encoding and an encoder block from tensor operations;
• train a small causal Transformer, inspect its attention and show what breaks when a design choice is removed;
• report results with evidence (multiple seeds, plots, a short analysis) rather than a single number.

### Ground rules

•   Python 3 and PyTorch. Layers such as Linear, LayerNorm and Embedding are allowed.
•   nn.LSTM, nn.GRU and nn.RNN may be used in the Problem 1 model comparison and experiments, but the LSTM cell in Problem 1 task 2 is written from scratch. nn.MultiheadAttention and nn.Transformer\* are not allowed anywhere in Problem 2. Library versions may be used only to check a student's own code.
•   No pre-trained weights.
•   A free Colab T4 GPU is enough: each Problem 1 run takes a few minutes and the Problem 2 language model takes about 10–15 minutes.
•	Frameworks. PyTorch only. TensorFlow, JAX and Keras are not accepted.
•	Data. Only the datasets named above (UCI HAR, Tiny Shakespeare and the generated adding problem). No external data and no pre-trained weights.
•	From-scratch parts. The RNN, LSTM and GRU cells (Problem 1 task 2) and the attention, multi-head attention and encoder block (Problem 2) must be the student's own code. Framework versions may be used only to check them.


### Environment.

• Python 3.10 or later, PyTorch 2.x, NumPy, pandas, Matplotlib and scikit-learn. List the exact versions in requirements.txt.
• Google Colab: Runtime → Change runtime type → T4 GPU, then check torch.cuda.is_available() returns True. Save results.csv after every run so a disconnected session loses nothing.
• Fix all seeds (random, numpy, torch). Small run-to-run differences on a GPU are expected and are not penalised.

### Problem 1 – RNN, LSTM and GRU on sequences (50 points)

You will build the three recurrent cells (vanilla RNN, LSTM, GRU) from their equations, then compare them on two sequence tasks: classifying real sensor sequences (UCI HAR) and a synthetic long-memory task (the adding problem) where the sequence length is a dial you control. The aim is to see, with evidence, why gated cells exist.
Data. (i) UCI HAR exactly as described under Datasets (128 steps × 9 channels, 6 classes, subject-wise split). (ii) The adding problem, generated in code (Dataset 3): each input has two channels, a random value and a marker; the target is the sum of the two marked values.
Tasks

1. Data pipelines (5). Build the HAR pipeline (subject-wise validation split, per-channel mean and std from the training part only, class-balance plot). Build the adding-problem generator with the provided function and write a test that every sequence has exactly two markers, one in each half, and that the target equals the sum of the marked values.
2. Cells from scratch (10). Implement MyRNNCell (tanh), MyLSTMCell and MyGRUCell using only matrix operations (no nn.RNN, nn.LSTM, nn.GRU or their cell classes inside these). Copy the weights from the library layer and show that your outputs match nn.RNN, nn.LSTM and nn.GRU within 1e-5 on a random batch. For d = 9 and h = 64 derive the parameter counts by hand (RNN 4,736; LSTM 18,944; GRU 14,208), reconcile them with your code, and explain why PyTorch reports 4,800, 19,200 and 14,400 (it keeps two bias vectors).
3. HAR classification (10). Train RNN, LSTM and GRU classifiers that read the 128 steps and classify from the last hidden state. Use the same hidden size, optimiser (Adam, lr 1e-3), batch size and epoch budget for all three, and 3 seeds each. Report test accuracy and macro-F1 (mean ± std), parameters and seconds per epoch, plus the confusion matrix of the best model.
4. Adding problem (10). For T = 50, 100, 200 and 400 train all three models with 3 seeds each. Report test MSE (mean ± std) against the constant-prediction baseline of about 0.167, and plot learning curves at T = 200. Call a run solved if test MSE is below 0.05, and tabulate how many runs per model and length are solved.
5. Gradients, ablations and analysis (15).
   o (a) At T = 100 plot ‖∂L/∂h_t‖ against t on a log scale for the three models, before and after training.
   o (b) Hidden-size sweep {16, 32, 64, 128} for LSTM and GRU on HAR.
   o (c) Individual ablation, chosen by the last digit of your student ID (see Individualisation): even = gradient clipping (norm 1.0 versus none) for the vanilla RNN at T = 200; odd = LSTM forget-gate bias initialised to 0 versus 1 at T = 200.
   o (d) One-page analysis: what the gradient plot says about memory, where each model fails as T grows, the accuracy–parameters–time trade-off, and which model you would put on a battery-powered wearable, with reasons taken from your tables.
   Sanity range for marking (approximate, not a target). On HAR, LSTM and GRU typically reach about 88–93% test accuracy, with the plain RNN lower and less stable. On the adding problem, short sequences are solved by all three; as T grows the plain RNN often stalls near the 0.167 baseline, while LSTM and GRU may need gradient clipping or more epochs at T = 200–400. Results outside these ranges are fine if the method is sound and the analysis explains them.

### Problem 2 – Transformer from scratch (50 points)

Students build attention and an encoder block from tensor operations, use them on two tasks, and measure what each design choice contributes.
Tasks

1. Attention from scratch (10 points). Write scaled dot-product attention with an optional mask. Unit tests must show that each row of weights sums to 1, that the output matches torch.nn.functional.scaled_dot_product_attention within 1e-5, and that a causal mask gives zero weight to future positions. Include a hand-checkable 3-token example supplied by the instructor, computed by hand and by code.
2. Multi-head attention and encoder block (10 points). Implement multi-head attention with W_Q, W_K, W_V and W_O and head reshaping, a pre-LayerNorm encoder block with a feed-forward layer of width 4d, residual connections and dropout, and sinusoidal positional encoding. Derive the weight count 4d² + 2·d·d_ff per block and match it to sum(p.numel()). For d = 128 and d_ff = 512 the weights alone come to 196,608 per block; students reconcile the biases and LayerNorm parameters.
3. Task A: encoder classifier on the HAR data (10 points). Reuse the Problem 1 data. Embed each time step (or each patch of 8 steps), add a CLS token, and classify with the encoder. Compare accuracy, parameters and training time with the best Problem 1 model. Ablate positional information: none, sinusoidal and learned.
4. Task B: character-level causal language model (10 points). Train on Tiny Shakespeare (about 1.1 MB, 65 distinct characters) with context 128, d = 128, 4 heads and 4 layers. Report validation loss and bits per character. Generate 500 characters at temperature 0.2, 0.8 and 1.5 with top-k sampling, and describe the differences.
5. Analysis and ablations (10 points). (a) Heatmaps of two heads on one sentence, with an interpretation. (b) Heads = 1, 4 and 8 at fixed d. (c) Remove the causal mask and show what happens to training loss and generation. (d) Measure time and memory at context 64, 128, 256 and 512 and relate the growth to O(n²·d).
   Sanity range for marking. A uniform guess over 65 characters gives a loss of ln 65 ≈ 4.17 nats. A small trained model typically reaches about 1.5–1.8 nats on validation. The range is approximate; the shape of the learning curve and the ablation evidence matter more.
   Optional stretch (up to 5 bonus points). Add a key-value cache to generation and report the speed-up, or train the Problem 1 LSTM on the same character task and compare it with the Transformer.

### Report and code requirements

The report is a PDF of at most 6 pages (A4, 11 pt, margins at least 2 cm). The AI-use statement and references do not count toward the 6 pages.
Report outline.

1. Title block: name, student ID
2. Problem 1: setup (datasets, split, preprocessing); verification of the RNN, LSTM and GRU cells (test result and parameter counts); HAR results table; adding-problem table and curves; experiments 5(a)–(c); the one-page analysis.
3. Problem 2: attention and encoder-block tests; parameter count derivation; Task A table; Task B learning curve and bits per character; heatmaps and ablations; the analysis.
4. AI-use statement and references.
   Every figure and table is numbered, has a caption, and is referred to in the text. A claim without a figure or table behind it earns no marks.
   Figures that must appear. Problem 1: class balance, confusion matrix of the best HAR model, adding-problem learning curves at T = 200 and MSE against T, gradient norm against time step, hidden-size sweep, the task 5(c) ablation plot. Problem 2: two attention heatmaps, learning curves, heads ablation, no-mask result, time and memory against context length.

### Code.

• Each notebook runs top to bottom with Restart and Run All and is submitted with its outputs saved.
• Unit tests are assert cells, placed directly after the function they test.
• Relative paths only; no paths such as /content/drive/MyDrive/... that exist only on the student's account.
• Comments explain the tensor shapes in the attention and LSTM code.
results.csv. One row per run, with exactly these columns: problem, model, seed, seq_len, test_acc, macro_f1, test_mse, params, sec_per_epoch, val_loss_nats, bits_per_char, notes. Columns that do not apply to a run are left empty. For the language model, bits_per_char is the validation loss in nats divided by ln 2.
generated_samples.txt. Three labelled samples of 500 characters, one each at temperature 0.2, 0.8 and 1.5.

### README.md, the one inside the folder dl-assignment-02/2026aj05102_DNN_Assignment.

At most one page: environment, how to run each notebook, the total run time on a T4, the student's variant, and anything that does not work.

### Marking rubric (100 points)- Scaled to standard marks after submission/evaluations

The assignment is out of 100 and can be scaled to whatever weightage it carries in the course. Marks go to method and evidence, not to the highest accuracy.

| #  | Task                                      | Points  | Full marks needs                                                                                                                 |
|----|-------------------------------------------|---------|----------------------------------------------------------------------------------------------------------------------------------|
| 1  | P1.1 Data pipelines                       | 5       | HAR subject-wise split, training-only statistics, class-balance plot; adding-problem generator with its test                     |
| 2  | P1.2 RNN, LSTM and GRU cells from scratch | 10      | All three cells match their library layers (e.g. nn.LSTM) within 1e-5; parameter counts explained, including the bias difference |
| 3  | P1.3 HAR classification                   | 10      | Same budget for all three, 3 seeds, mean ± std, parameters, time and confusion matrix reported                                   |
| 4  | P1.4 Adding problem                       | 10      | T = 50, 100, 200, 400 × 3 models × 3 seeds; MSE against the 0.167 baseline; solved-run table; curves at T = 200                  |
| 5  | P1.5 Gradients, ablations and analysis    | 15      | Gradient-norm plot, hidden-size sweep, individual ablation; conclusions backed by the tables; wearable recommendation justified  |
| 6  | P2.1 Attention from scratch               | 10      | Tests pass; hand example and code agree                                                                                          |
| 7  | P2.2 Multi-head and encoder block         | 10      | Correct shapes; parameter count derived and reconciled with code                                                                 |
| 8  | P2.3 Encoder classifier on HAR            | 10      | Fair comparison with Problem 1; positional-encoding ablation                                                                     |
| 9  | P2.4 Character language model             | 10      | Learning curve, bits per character, three temperatures discussed                                                                 |
| 10 | P2.5 Analysis and ablations               | 10      | Heatmaps interpreted; heads, no-mask and context-length studies                                                                  |
| 11 | Bonus                                     | up to 5 | Key-value cache or LSTM-versus-Transformer character model                                                                       |

Bonus up to 5 Key-value cache or LSTM-versus-Transformer character model
Deductions. Up to 10 points off for code that does not run from the README, results that cannot be reproduced from the fixed seeds, or a report over 6 pages. Copied code, or use of a forbidden layer in a from-scratch part, is handled under the academic-integrity policy.

### FAQ
