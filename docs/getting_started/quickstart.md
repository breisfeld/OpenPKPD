# Quick Start

This page runs through a complete FOCE estimation in under 20 lines of Python.

## The Python API

```python
from openpkpd import ModelBuilder

built = (
    ModelBuilder()
    .problem("Theophylline 1-cmt oral")
    .data("theo.csv")                       # CSV file path
    .subroutines(advan=2, trans=2)          # 1-cmt oral, CL/V parameterisation
    .pk("""
        KA = THETA(1) * EXP(ETA(1))
        CL = THETA(2) * EXP(ETA(2))
        V  = THETA(3) * EXP(ETA(3))
    """)
    .error("Y = F * (1 + EPS(1))")         # Proportional residual error
    .theta([(0.01, 1.5, 20),               # KA: (lower, init, upper)
            (0.001, 0.08, 5),              # CL
            (0.1, 30, 500)])               # V
    .omega([0.5, 0.3, 0.3])               # Diagonal OMEGA (variances)
    .sigma(0.1)                            # SIGMA
    .estimation(method="FOCE", interaction=True, maxeval=9999)
    .covariance()                          # Enable covariance step
    .build()
)

result = built.fit()
print(result.summary())
```

### Reading the result

```python
result.ofv              # Final objective function value (−2 log-likelihood)
result.theta_final      # np.ndarray of final THETA estimates
result.omega_final      # Final OMEGA matrix
result.sigma_final      # Final SIGMA matrix
result.converged        # True if optimizer converged
result.post_hoc_etas    # {subject_id: eta_vector} empirical Bayes estimates
result.eta_shrinkage    # ETA shrinkage per random effect
result.ofv_history      # OFV at each outer iteration

result.compute_shrinkage()   # Populate eta_shrinkage from post_hoc_etas
print(result.summary())      # One-line text summary
```

### Writing HTML and PDF reports

```python
from openpkpd import export_html_report_to_pdf, write_pdf_report

result.to_html("report.html", params=built.params, title="Theophylline FOCE")
result.to_pdf("report.pdf", params=built.params, title="Theophylline FOCE")
write_pdf_report("report-copy.pdf", result, built.params, title="Theophylline FOCE")
export_html_report_to_pdf("report.html", "report-from-html.pdf")
```

PDF export uses the optional Qt-based GUI runtime, so install
`openpkpd[gui]` when you want `result.to_pdf(...)`, `write_pdf_report(...)`,
or `export_html_report_to_pdf(...)`.

## Running from a control stream file

```bash
openpkpd run model.ctl
openpkpd run model.ctl --method FOCE --verbose
openpkpd parse model.ctl --json      # Inspect parsed records
```

```python
from openpkpd.parser.control_stream import ControlStream
cs = ControlStream.from_file("model.ctl")
print(cs.problem.title)
print(cs.estimation_records[0].method)
```

## Desktop GUI quick start

Install the GUI extra and launch the application:

```bash
pip install "OpenPKPD[gui]"
openpkpd-gui
```

The GUI is organized around a **workspace / project / scenario** tree. In a new
session, the fastest path is usually:

1. create a project and scenario from **Workspace**
2. import a CSV in **Data**
3. author or open a model in **Model**
4. run estimation in **Fit**
5. review outputs in **Overview**, **Results**, **Plots**, and **Diagnostics**

The **Advanced** workflow adds tabbed post-fit tools for **VPC**,
**Bootstrap**, **Design**, and **Artifacts**.

## Diagnostic plots

```python
from openpkpd.plots.diagnostics import compute_diagnostics
from openpkpd.plots.gof import diagnostic_panel
from openpkpd.plots.pk import spaghetti_plot

diag_df = compute_diagnostics(built.population_model, result)

fig = diagnostic_panel(diag_df, title="My model — GOF")
fig.savefig("gof_panel.png", dpi=150)

fig2 = spaghetti_plot(diag_df)
```

:::{tip}
Plots require `matplotlib`. Install with `uv add "OpenPKPD[plots]"`.
:::

## What's next?

- {doc}`/user_guide/model_builder` — every `ModelBuilder` method explained
- {doc}`/user_guide/gui` — the current desktop GUI layout, menus, and workflows
- {doc}`/user_guide/pk_subroutines` — choosing the right ADVAN and TRANS
- {doc}`/user_guide/estimation_methods` — FO vs FOCE vs SAEM
- {doc}`/examples/index` — seven annotated worked examples
