# Finke-Watzky Model — Fitting to Published Data

**Course:** Physics of Molecular Diseases — Week 1  
**Question 6:** What to do if you want to fit your model to published data? How to quickly obtain data from article graphs? Digitize published data using WebPlotDigitizer and fit the F-W model.

---

## Overview

This script demonstrates the full workflow for extracting quantitative kinetic parameters ($k_1$, $k_2$) from published protein aggregation figures:

1. **Digitize** the figure using WebPlotDigitizer
2. **Load** the CSV output into Python
3. **Estimate** initial parameter guesses automatically from the data shape
4. **Fit** the F-W analytical solution using `scipy.optimize.curve_fit`
5. **Extract** $k_1$ and $k_2$ and compare with the paper's reported values
6. **Assess** goodness of fit via $R^2$

Applied to Figure 2 (amyloid-$\beta$, Kelly et al.) and Figure 7 ($\alpha$-synuclein, Fink et al.) of Morris et al. (2008).

---

## WebPlotDigitizer Workflow

[WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/) is a free web app for extracting numerical data from published figures.

### Step-by-step

**1. Prepare the figure**  
Take a screenshot or save the figure from the paper as PNG/JPG.

**2. Upload and select axis type**  
Go to https://automeris.io/WebPlotDigitizer/ → *Load Image* → select "XY" for standard scatter/line plots.

**3. Calibrate the axes**  
Click exactly 4 points on the axes (2 on X, 2 on Y) and enter their known values. Example for Figure 2 of Morris et al.:
- X1 = 0 h, X2 = 150 h
- Y1 = 0 (0% aggregated), Y2 = 1 (100% aggregated)

**4. Pick data points**  
Either click each point manually (*Add Point*) or use automatic detection (*Automatic Extraction*) for dense datasets.

**5. Export as CSV**  
*File → Export → CSV* → save as `(...).csv`

### Load in Python

```python
import pandas as pd
import numpy as np

data = pd.read_csv("fig2_abeta.csv")
t_data = data.iloc[:, 0].values          # time column
B_data = data.iloc[:, 1].values          # signal column

# Normalise to [0, 1] if data is in % or arbitrary units
B_data = B_data / B_data.max()
```

---

## Fitting the F-W Model

### Fitting function

The normalised F-W analytical solution (eq. 3, Morris et al. 2008) is used as the fitting function:

$$[B]_t = [A]_0 - \frac{\dfrac{k_1}{k_2} + [A]_0}{1 + \dfrac{k_1}{k_2[A]_0} \exp\!\bigl((k_1 + k_2[A]_0)\,t\bigr)}$$

For normalised data ($[A]_0 = 1$), this simplifies to `fw_B_normalised(t, k1, k2)`.

### Automatic p0 estimation

A key improvement over naive fitting: initial guesses are estimated **directly from the data shape** rather than set by hand. This prevents `curve_fit` from getting stuck in a flat local minimum.

```python
def estimate_p0(t_data, B_data):
    # k1 ≈ 1/t_lag  (time to reach 5% aggregation)
    above_5 = t_data[B_data >= 0.05]
    t_lag = above_5[0] if len(above_5) > 0 else t_data[len(t_data)//2]
    k1_guess = 1.0 / t_lag

    # k2 ≈ slope at inflection point (where B ≈ 0.5)
    idx = np.argmin(np.abs(B_data - 0.5))
    slope = (B_data[idx+1] - B_data[idx-1]) / (t_data[idx+1] - t_data[idx-1])
    k2_guess = max(slope, 1e-4)

    return [k1_guess, k2_guess]
```

This works for any normalised dataset without manual tuning.

### Running the fit

```python
from scipy.optimize import curve_fit

p0 = estimate_p0(t_data, B_data)

popt, pcov = curve_fit(
    fw_B_normalised,
    t_data, B_data,
    p0=p0,
    bounds=([1e-6, 1e-4], [10.0, 100.0]),   # wide, physically meaningful
    maxfev=100_000
)
k1_fit, k2_fit = popt
k1_err, k2_err = np.sqrt(np.diag(pcov))
```

### Important note on bounds

Using bounds that are too tight is a common cause of fitting failure — the curve appears flat near zero because `curve_fit` cannot explore the right parameter range. The bounds used here are intentionally wide:

```python
# WRONG — too tight for normalised data, causes flat curve:
bounds = ([1e-8, 1e-5], [1e-2, 1.0])

# CORRECT — wide enough for normalised [0,1] data:
bounds = ([1e-6, 1e-4], [10.0, 100.0])
```

### Goodness of fit

```python
def r_squared(y_data, y_fit):
    ss_res = np.sum((y_data - y_fit) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    return 1.0 - ss_res / ss_tot

R2 = r_squared(B_data, fw_B_normalised(t_data, k1_fit, k2_fit))
# Morris et al. report R² ≥ 0.98 for all 14 datasets
```

---

## Results

| Dataset | $k_1$ ($h^{-1}$, norm.) | $R^2$ | Paper's $k_1$ ($h^{-1}$) |
|---------|------------------------|-------|--------------------------|
| Amyloid- $\beta$ (Fig. 2) | $3.4\times10^{-5}$ | 0.9993 | $8\times10^{-6}$ |
| $\alpha$-synuclein (Fig. 7) | $5.1\times10^{-5}$ | 0.9985 | $4\times10^{-5}$ |

Both fits exceed $R^2 = 0.998$, consistent with the paper's $R^2 \geq 0.98$.

### Note on units of $k_2$

The paper reports $k_2$ in $\mu$ $M^{-1}$ $h^{-1}$ (absolute concentration). When fitting normalised data ($[A]_0 = 1$, dimensionless), the fitted $k_2$ absorbs the concentration scaling and **cannot be directly compared** to the paper's values. Only $k_1$ (units $h^{-1}$) is directly comparable.

To recover the paper's $k_2$:

```python
k2_abs = k2_fit / A0_concentration   # divide by [A]0 in µM
```

---

## Figures

**Figure 1 (2×2 layout):**
- Top row: full timecourse (0–170 h) with lag / growth / plateau phases shaded
- Bottom row: zoom into the transition region where the sigmoidal rise occurs

The 2×2 layout is essential because the lag phase dominates the full timecourse view, making the sigmoidal transition appear compressed. The zoom panel shows the shape clearly.

**Figure 2:** Residuals (data − fit) for both datasets. Random scatter around zero confirms no systematic error in the fit.

---

## Practical tips for curve_fit

| Issue | Solution |
|-------|----------|
| Curve appears flat near zero | Use `estimate_p0()` — bounds were probably too tight |
| Fit does not converge | Check data normalisation; widen bounds |
| $R^2$ is low | More data points around the transition region help |
| Large parameter uncertainties | The sigmoidal transition needs to be well-sampled |

---

## Dependencies

```
numpy
scipy
matplotlib
pandas  (optional, for loading CSV)
```

---

## Key References
- Prof. Ala Trusina, Lecture notes from the course "Physics of Molecular Diseases", Niels Bohr Institute, 2020
- Morris et al. (2008) — original F-W fitting paper, *Biochemistry* 47, 2413–2427  
- WebPlotDigitizer — Rohatgi, A. (2021), https://automeris.io/WebPlotDigitizer/  
- `scipy.optimize.curve_fit` — https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
