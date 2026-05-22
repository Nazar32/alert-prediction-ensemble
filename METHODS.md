# Methods Reference

Technical descriptions of every component used in the reference (DNN+RF) instantiation of the per-region precision-constrained calibration framework. The framework's primary contribution — **per-region threshold calibration** — is documented in §16; the surrounding components (DNN, RF, focal loss, isotonic recalibration) are part of the reference instantiation and are interchangeable with any probabilistic classifier under the same calibration scope.

---

## 1. Fully Connected (Linear) Layer

A linear transformation `y = Wx + b` where `W` is a weight matrix and `b` is a bias vector. Each neuron in the layer computes a weighted sum of all inputs from the previous layer. The four layers in the DNN progressively compress the 95-dimensional input: 95 → 256 → 128 → 64 → 1.

---

## 2. ReLU Activation

Rectified Linear Unit: `f(x) = max(0, x)`. Applied after each hidden linear layer. ReLU introduces non-linearity (without which stacking linear layers would collapse to a single linear transform) and avoids the vanishing-gradient problem that affects sigmoid/tanh activations in deep networks, because its gradient is a constant 1 for positive inputs.

---

## 3. Batch Normalisation (BatchNorm)

Applied after the first two linear layers. For each mini-batch, BatchNorm normalises the layer's output to zero mean and unit variance, then applies learned scale (γ) and shift (β) parameters. This:
- reduces internal covariate shift, making training faster and more stable,
- acts as a mild regulariser,
- allows higher learning rates without divergence.

The normalisation statistics (mean and variance) are computed over the batch during training and replaced by running averages at inference time.

---

## 4. Dropout

A regularisation technique that randomly sets a fraction `p` of neuron outputs to zero during each training forward pass (the remaining outputs are scaled by `1/(1−p)` to preserve expected magnitude). This prevents co-adaptation — neurons cannot rely on specific other neurons always being present — which reduces overfitting.

Dropout rates used:
- `0.2` after layers 1 and 2 (higher regularisation in the wider layers),
- `0.1` after layer 3 (lighter regularisation near the output).

Dropout is disabled at inference time (`model.eval()`).

---

## 5. Sigmoid Output Activation

`σ(x) = 1 / (1 + e^{−x})`. Squashes the scalar output of the final linear layer into the range (0, 1), making it interpretable as a probability `p̂(alert = 1 | x)`. This probability is what gets blended with the Random Forest output and subsequently calibrated.

---

## 6. Xavier (Glorot) Initialisation

Weight matrices are initialised by drawing from a uniform distribution scaled by `sqrt(6 / (fan_in + fan_out))`, where `fan_in` and `fan_out` are the number of input and output units of the layer. This keeps the variance of activations approximately constant across layers at the start of training, avoiding vanishing or exploding gradients before any learning has occurred. Biases are initialised to zero.

---

## 7. Binary Focal Loss

Introduced by Lin et al. (2017) for object detection with extreme foreground/background imbalance; directly applicable to rare event prediction.

Standard binary cross-entropy treats every example equally:

```
BCE(p, y) = −[y·log(p) + (1−y)·log(1−p)]
```

Focal loss adds a modulating factor `(1 − pₜ)^γ` where `pₜ` is the model's predicted probability for the *correct* class:

```
FL(p, y) = −αₜ · (1 − pₜ)^γ · log(pₜ)
```

- **γ (focusing exponent) = 2.0** — examples where the model is already confident (`pₜ → 1`) receive near-zero weight; the gradient focuses on hard, uncertain examples near the decision boundary.
- **α (class weight) = 0.60** — the positive class (alert active) receives 60% of the total weight, the negative class 40%. This counteracts the ~18% positive rate in the dataset.

Together, γ and α make the DNN more conservative about issuing positive predictions, which directly raises precision.

---

## 8. Adam Optimiser

Adaptive Moment Estimation. Maintains per-parameter running estimates of the first moment (mean of gradients) and second moment (mean of squared gradients), and uses these to compute an adaptive learning rate for each parameter:

```
mₜ = β₁·mₜ₋₁ + (1−β₁)·gₜ          (first moment)
vₜ = β₂·vₜ₋₁ + (1−β₂)·gₜ²         (second moment)
θₜ = θₜ₋₁ − lr · m̂ₜ / (√v̂ₜ + ε)
```

Parameters used: `lr = 1e-3`, `weight_decay = 1e-4` (L2 regularisation added to the loss), default β₁ = 0.9, β₂ = 0.999.

---

## 9. L2 Weight Decay (via Adam `weight_decay`)

Adds a penalty `(λ/2) · ||W||²` to the loss. Encourages the network to keep weight magnitudes small, which penalises overly complex solutions and reduces overfitting. Here `λ = 1e-4`.

---

## 10. Gradient Clipping

Before each parameter update, the global norm of all gradients is computed. If it exceeds `max_norm = 1.0`, all gradients are rescaled proportionally so the norm equals 1.0. This prevents "exploding gradients" — occasional very large gradient updates that can destabilise training, particularly in combination with focal loss on hard examples.

---

## 11. Cosine Annealing Learning Rate Schedule

The learning rate follows a cosine curve from its initial value (`lr = 1e-3`) down to near zero over `T_max = epochs × steps_per_epoch` total gradient steps. This "warm" decay allows the optimiser to take large steps early (fast convergence) and fine-grained steps later (better final minimum), without requiring manual learning-rate tuning.

---

## 12. Early Stopping

After each epoch, ROC-AUC is evaluated on the validation set. If validation AUC does not improve for `patience = 15` consecutive epochs, training stops and the weights from the best epoch are restored. This prevents overfitting to the training distribution without requiring a fixed number of epochs.

---

## 13. Random Forest

An ensemble of `n_estimators = 300` decision trees trained independently on bootstrap samples of the training data (bagging). Each split in each tree considers only `max_features = sqrt(d)` randomly chosen features. Predictions are the average of all tree probability estimates.

Parameters used:
- `max_depth = 20` — limits tree complexity to reduce overfitting,
- `min_samples_split = 5`, `min_samples_leaf = 2` — minimum sample counts before a split is allowed,
- `class_weight = 'balanced'` — sample weights are inversely proportional to class frequency, compensating for the 18% positive rate without modifying the data.

Random Forests are low-variance and handle heterogeneous feature types (binary lags, continuous rolling stats, one-hot dummies) without scaling, complementing the DNN which requires z-scored inputs.

---

## 14. α-Weighted Ensemble Blending

The final ensemble probability is a convex combination of the DNN and RF outputs:

```
p̂_ens = α · p̂_NN + (1 − α) · p̂_RF
```

The blend weight `α` is searched over 19 candidates {0.05, 0.10, …, 0.95} by evaluating validation F1 at each candidate. The best value found is `α ≈ 0.60`, giving the DNN a slight majority. Blending reduces variance compared to either model alone: errors that are uncorrelated across the two models cancel out.

---

## 15. Isotonic Regression (Probability Calibration)

Focal loss with γ > 0 compresses predicted probabilities away from 0 and 1 — the raw outputs cluster in the middle of [0, 1] rather than being well-calibrated. Isotonic regression fits a non-decreasing step function mapping raw probabilities to calibrated ones, minimising mean squared error on the validation set. This makes the probabilities more reliable as thresholds are applied later.

```python
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p_ens_val, y_val)
p_calibrated = iso.transform(p_ens_test)
```

---

## 16. Per-Region Threshold Calibration

**The framework's primary contribution.** The default decision threshold of 0.5 is rarely optimal, especially under class imbalance with heterogeneous per-region base rates. For each of the 25 regions, a separate threshold `τ(r)` is chosen by scanning 199 candidates in [0.01, 0.99] on the validation set and selecting:

```
τ(r) = argmax F1   subject to   Precision(r) ≥ 75%
```

This encodes the operational requirement (most issued alerts must be real) directly into the decision boundary. The calibrated thresholds are **inversely related to alert frequency**: low-frequency western regions receive higher thresholds (Zakarpattia: 0.856, Volyn: 0.802) because the classifier must be more selective to satisfy the precision floor when positives are rare, while high-frequency eastern regions receive lower thresholds (Donetsk: 0.337, Sumy: 0.446) because the strong autocorrelation signal lets a lower threshold still meet the floor with high recall.

If a region has fewer than 30 **positive** validation samples, or if no threshold satisfies the precision floor, the region falls back to a global threshold calibrated on all validation data (`τ_global = 0.3763` on the Ukrainian dataset).

The cross-model experiments (LightGBM, XGBoost, LSTM in addition to DNN+RF) demonstrate that this per-region calibration step contributes +3–4 percentage points precision **independently of the choice of base classifier**, making it the dominant design decision for spatially heterogeneous rare-event prediction under precision constraints.

---

## 17. Z-Score Normalisation (StandardScaler)

All 70 continuous features are standardised to zero mean and unit standard deviation using statistics computed on the training set only:

```
x_scaled = (x − μ_train) / σ_train
```

The 25 one-hot oblast dummies are passed unscaled (they are already in {0, 1}). Normalisation prevents features with large numerical ranges (e.g. rolling counts over 72 h) from dominating gradient updates in the DNN, and helps Adam converge faster.

---

## References

- **Focal Loss**: Lin, T.-Y. et al. (2017). *Focal Loss for Dense Object Detection*. ICCV.
- **Batch Normalisation**: Ioffe, S. & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training*. ICML.
- **Dropout**: Srivastava, N. et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR.
- **Xavier Initialisation**: Glorot, X. & Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward neural networks*. AISTATS.
- **Adam**: Kingma, D. P. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR.
- **Cosine Annealing**: Loshchilov, I. & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts*. ICLR.
- **Random Forests**: Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
- **Isotonic Regression Calibration**: Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting Good Probabilities with Supervised Learning*. ICML.
