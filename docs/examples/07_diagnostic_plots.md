# Example 7 — Full Diagnostic Plot Gallery

**Script:** `examples/07_diagnostic_plots.py`

A complete walkthrough of every diagnostic plot function using the
theophylline FO model from {doc}`01_theophylline_fo`.

## Output

```{literalinclude} ../_static/examples/07_output.txt
:language: text
```

## Figures

### GOF panel (2×3 composite)

The panel contains DV vs IPRED, DV vs PRED, CWRES vs TIME, CWRES vs PRED,
CWRES Q-Q, and |IWRES| vs IPRED.

![GOF panel](../_static/examples/07_gof_panel.png)

### Individual GOF plots

![DV vs IPRED](../_static/examples/07_dv_vs_ipred.png)
![DV vs PRED](../_static/examples/07_dv_vs_pred.png)
![CWRES vs TIME](../_static/examples/07_cwres_time.png)
![CWRES vs PRED](../_static/examples/07_cwres_pred.png)
![CWRES Q-Q](../_static/examples/07_cwres_qq.png)
![|IWRES| vs IPRED](../_static/examples/07_abs_iwres.png)

### PK concentration-time plots

![Spaghetti plot](../_static/examples/07_spaghetti.png)
![Concentration-time](../_static/examples/07_conc_time.png)
![Mean profile](../_static/examples/07_mean_profile.png)

### ETA diagnostics

![ETA histograms](../_static/examples/07_eta_hist.png)
![ETA pairs](../_static/examples/07_eta_pairs.png)
![ETA1 vs weight](../_static/examples/07_eta1_vs_wt.png)

### OFV convergence

![OFV history](../_static/examples/07_ofv_history.png)

## Interpreting GOF plots

| Plot | What to look for |
|------|-----------------|
| DV vs IPRED / PRED | Points close to the identity line, no systematic bias |
| CWRES vs TIME | Random scatter around zero; no trend or funnel |
| CWRES vs PRED | Random scatter; no heteroscedasticity |
| CWRES Q-Q | Points on the diagonal line; normality of residuals |
| \|IWRES\| vs IPRED | Homogeneous spread; no increasing trend (proportional error check) |
| ETA histograms | Symmetric, approximately normal distributions |
| ETA pairs | No strong correlations (would suggest OMEGA mis-specification) |
| ETA vs covariates | Flat relationship (no unexplained covariate effect) |
