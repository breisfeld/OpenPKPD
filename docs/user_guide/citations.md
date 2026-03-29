# Algorithm and Method Citations

This document provides bibliographic references for the statistical methods,
pharmacokinetic models, and numerical algorithms implemented in OpenPKPD.
Entries are grouped by functional area. Where a method is well-established
(e.g. the one-compartment model) and is not attributable to a single primary
paper, the most commonly cited textbook or reference implementation is given
instead.

---

## Estimation methods

### First-Order (FO) and First-Order Conditional Estimation (FOCE/FOCEI)

The linearisation-based likelihood approximations underlying FO and FOCE were
developed during the original NONMEM project:

1. Sheiner LB, Rosenberg B, Marathe VV (1977). Estimation of population
   characteristics of pharmacokinetic parameters from routine clinical data.
   *J Pharmacokinet Biopharm* **5**(5):445–479.

2. Sheiner LB, Beal SL (1980). Evaluation of methods for estimating population
   pharmacokinetic parameters I. Michaelis-Menten model: routine clinical
   pharmacokinetic data. *J Pharmacokinet Biopharm* **8**(6):553–571.

3. Sheiner LB, Beal SL (1983). Evaluation of methods for estimating population
   pharmacokinetic parameters III. Monoexponential model: routine clinical
   pharmacokinetic data. *J Pharmacokinet Biopharm* **11**(3):303–319.

4. Beal SL, Sheiner LB (1992). *NONMEM Users Guides*. University of
   California, San Francisco.

FOCEI (FOCE with interaction) adds ETA–EPS interaction to correctly account
for proportional or combined residual error models:

5. Bauer RJ (2019). NONMEM tutorial part I: description of commands and
   options, with simple examples of population analysis.
   *CPT Pharmacometrics Syst Pharmacol* **8**(8):525–537.

### Laplace Approximation

6. Tierney L, Kadane JB (1986). Accurate approximations for posterior moments
   and marginal densities. *J Am Stat Assoc* **81**(393):82–86.

7. Wolfinger R (1993). Laplace's approximation for nonlinear mixed models.
   *Biometrika* **80**(4):791–795.

### SAEM (Stochastic Approximation EM)

The stochastic EM algorithm with decaying step-size and Rao–Blackwellised
sufficient statistics:

8. Delyon B, Lavielle M, Moulines E (1999). Convergence of a stochastic
   approximation version of the EM algorithm. *Ann Stat* **27**(1):94–128.

9. Kuhn E, Lavielle M (2004). Coupling a stochastic approximation version of
   EM with an MCMC procedure. *ESAIM Probab Stat* **8**:115–131.

10. Kuhn E, Lavielle M (2005). Maximum likelihood estimation in nonlinear
    mixed effects models. *Comput Stat Data Anal* **49**(4):1020–1038.

### IMP / IMPMAP (Importance Sampling)

11. Comets E, Lavenu A, Lavielle M (2017). Parameter estimation in nonlinear
    mixed effect models using saemix, an R implementation of the SAEM algorithm.
    *J Stat Softw* **80**(3):1–41. *(describes the importance-sampling
    likelihood evaluation used in IMP mode)*

12. Lavielle M (2014). *Mixed Effects Models for the Population Approach:
    Models, Tasks, Methods and Tools*. CRC Press, Boca Raton.
    *(Chapter 7 covers importance sampling for marginal likelihood evaluation.)*

### Bayesian (MCMC / NUTS)

13. Gelman A, Carlin JB, Stern HS, Dunson DB, Vehtari A, Rubin DB (2013).
    *Bayesian Data Analysis*, 3rd ed. CRC Press, Boca Raton.

14. Hoffman MD, Gelman A (2014). The No-U-Turn sampler: adaptively setting path
    lengths in Hamiltonian Monte Carlo. *J Mach Learn Res* **15**:1593–1623.

15. Carpenter B, Gelman A, Hoffman MD, et al. (2017). Stan: a probabilistic
    programming language. *J Stat Softw* **76**(1):1–32.

### Nonparametric (NPML/NPEM)

16. Mallet A (1986). A maximum likelihood estimation method for random
    coefficient regression models. *Biometrika* **73**(3):645–656.

17. Schumitzky A (1991). Nonparametric EM algorithms for estimating prior
    distributions. *Appl Math Comput* **45**(2–3):143–157.

### Empirical Bayes Estimates (post-hoc etas)

18. Sheiner LB, Beal SL (1982). Bayesian individualization of pharmacokinetics:
    simple implementation and comparison with non-Bayesian methods.
    *J Pharm Sci* **71**(12):1344–1348.

---

## Covariance step

### Sandwich (R⁻¹SR⁻¹) Estimator

19. White H (1982). Maximum likelihood estimation of misspecified models.
    *Econometrica* **50**(1):1–25.

20. Beal SL (1992). NONMEM Users Guide — Part VII: Supplemental User's
    Guide. University of California, San Francisco.
    *(describes the R and S matrices and the sandwich covariance formula.)*

---

## PK structural models (ADVAN subroutines)

The ADVAN/TRANS parameterisation and compartmental model notation follows the
NONMEM convention:

21. Boeckmann AJ, Sheiner LB, Beal SL (1992). *NONMEM Users Guide — Part V:
    Introductory Guide*. University of California, San Francisco.

### Bateman function (ADVAN2 — one-compartment oral)

The one-compartment oral absorption biexponential solution (sometimes called the
Bateman function) follows from the standard linear ODE system and is treated as
a textbook result:

22. Rescigno A, Segre G (1966). *Drug and Tracer Kinetics*. Blaisdell,
    Waltham. *(classical derivation of the one-compartment oral solution.)*

23. Gibaldi M, Perrier D (1982). *Pharmacokinetics*, 2nd ed. Marcel Dekker,
    New York. *(standard reference for multi-compartment analytical solutions.)*

### Transit compartment absorption (ADVAN with transit chain)

24. Savic RM, Jonker DM, Kerbusch T, Karlsson MO (2007). Implementation of a
    transit compartment model for describing drug absorption in
    pharmacokinetic studies. *J Pharmacokinet Pharmacodyn* **34**(5):711–726.

### Michaelis–Menten elimination (ADVAN10)

25. Michaelis L, Menten ML (1913). Die Kinetik der Invertinwirkung.
    *Biochem Z* **49**:333–369. *(foundational paper.)*

26. Wagner JG (1973). Properties of the Michaelis-Menten equation and its
    integrated form which are useful in pharmacokinetics.
    *J Pharmacokinet Biopharm* **1**(2):103–121.

### ODE solvers (ADVAN6/8/13)

The underlying solvers are from SciPy:

27. Virtanen P, Gommers R, Oliphant TE, et al. (2020). SciPy 1.0:
    fundamental algorithms for scientific computing in Python.
    *Nat Methods* **17**(3):261–272.

### Delay differential equations (ADVAN16-style DDE)

29. Bellman R, Cooke KL (1963). *Differential-Difference Equations*.
    Academic Press, New York. *(foundational DDE theory.)*

30. Shampine LF, Thompson S (2001). Solving DDEs in Matlab.
    *Appl Numer Math* **37**(4):441–458. *(step-size control and dense
    output for DDEs, on which the OpenPKPD DDE integrator is based.)*

---

## Pharmacodynamic models

### Emax and Hill (sigmoidal Emax) models

31. Hill AV (1910). The possible effects of the aggregation of the molecules
    of haemoglobin on its dissociation curves. *J Physiol*
    **40**(Suppl):iv–vii. *(original cooperativity equation.)*

32. Holford NHG, Sheiner LB (1981). Understanding the dose-effect
    relationship: clinical application of pharmacokinetic-pharmacodynamic
    models. *Clin Pharmacokinet* **6**(6):429–453.

### Effect compartment (link model)

33. Sheiner LB, Stanski DR, Vozeh S, Miller RD, Ham J (1979). Simultaneous
    modeling of pharmacokinetics and pharmacodynamics: application to
    d-tubocurarine. *Clin Pharmacol Ther* **25**(3):358–371.

### Indirect response models (IDR types I–IV)

34. Dayneka NL, Garg V, Jusko WJ (1993). Comparison of four basic models of
    indirect pharmacodynamic responses. *J Pharmacokinet Biopharm*
    **21**(4):457–478.

35. Turnin JE, Peck CC, Sheiner LB (1995). Pharmacokinetic-pharmacodynamic
    modeling in drug development. *Annu Rev Pharmacol Toxicol*
    **35**:497–520.

### Target-mediated drug disposition (TMDD)

36. Levy G (1994). Pharmacologic target-mediated drug disposition.
    *Clin Pharmacol Ther* **56**(3):248–252. *(coined the concept.)*

37. Mager DE, Jusko WJ (2001). General pharmacokinetic model for drugs
    exhibiting target-mediated drug disposition. *J Pharmacokinet
    Pharmacodyn* **28**(6):507–532. *(full TMDD ODE system.)*

38. Gibiansky L, Gibiansky E, Kakkar T, Ma P (2008). Approximations of the
    target-mediated drug disposition model and identifiability of model
    parameters. *J Pharmacokinet Pharmacodyn* **35**(5):573–591.
    *(QSS and Michaelis–Menten approximations.)*

### Tumor growth inhibition (TGI)

39. Simeoni M, Magni P, Cammia C, De Nicolao G, Croci V, Pesenti E,
    Germani M, Poggesi I, Rocchetti M (2004). Predictive pharmacokinetic-
    pharmacodynamic modeling of tumor growth kinetics in xenograft models
    after administration of anticancer agents.
    *Cancer Res* **64**(3):1094–1101.

### Time-to-event (TTE) / survival models

40. Cox DR (1972). Regression models and life-tables (with discussion).
    *J R Stat Soc B* **34**(2):187–220. *(foundational survival analysis.)*

41. Holford NHG (2013). A time to event tutorial for pharmacometricians.
    *CPT Pharmacometrics Syst Pharmacol* **2**(5):e43.

### Count and categorical PD

42. Agresti A (2002). *Categorical Data Analysis*, 2nd ed. Wiley, Hoboken.
    *(proportional odds model and related categorical methods.)*

43. Plan EL (2014). Modeling and simulation of count data.
    *CPT Pharmacometrics Syst Pharmacol* **3**(8):e129.

---

## Covariate modelling

### Stepwise covariate modelling (SCM)

44. Jonsson EN, Karlsson MO (1998). Automated covariate model building within
    NONMEM. *Pharm Res* **15**(9):1463–1468.

### Covariate effect parameterisations

Standard power-law and linear covariate models are textbook content; the
allometric scaling exponent for clearance (0.75) is widely attributed to:

45. Anderson BJ, Holford NHG (2008). Mechanism-based concepts of size and
    maturity in pharmacokinetics. *Annu Rev Pharmacol Toxicol* **48**:303–332.

---

## Non-compartmental analysis (NCA)

### AUC by the trapezoidal / log-linear rule

46. Yeh KC, Kwan KC (1978). A comparison of numerical integrating algorithms
    by trapezoidal, Lagrange, and spline approximation. *J Pharmacokinet
    Biopharm* **6**(1):79–98.

47. Purves RD (1992). Optimum numerical integration methods for estimation of
    area-under-the-curve (AUC) and area-under-the-moment-curve (AUMC).
    *J Pharmacokinet Biopharm* **20**(3):211–226.

### Terminal elimination rate constant (λ_z) and half-life

48. Gabrielsson J, Weiner D (2006). *Pharmacokinetic and Pharmacodynamic Data
    Analysis: Concepts and Applications*, 4th ed. Swedish Pharmaceutical
    Press, Stockholm. *(standard NCA textbook reference.)*

### Bioequivalence — two one-sided t-tests (TOST)

49. Schuirmann DJ (1987). A comparison of the two one-sided tests procedure
    and the power approach for assessing the equivalence of average
    bioavailability. *J Pharmacokinet Biopharm* **15**(6):657–680.

50. U.S. Food and Drug Administration (2001). *Statistical Approaches to
    Establishing Bioequivalence*. FDA, Rockville.

---

## Simulation and diagnostics

### Visual predictive check (VPC)

51. Karlsson MO, Holford N (2008). A tutorial on visual predictive checks.
    PAGE 17, Abstr 1434. www.page-meeting.org/?abstract=1434.

### Prediction-corrected VPC (pcVPC)

52. Bergstrand M, Hooker AC, Wallin JE, Karlsson MO (2011).
    Prediction-corrected visual predictive checks for diagnosing nonlinear
    mixed-effects models. *AAPS J* **13**(2):143–151.

### Normalised prediction distribution error (NPDE)

53. Brendel K, Comets E, Laffont C, Laveille C, Mentré F (2006).
    Metrics for external model evaluation with an application to the
    population pharmacokinetics of gliclazide. *Pharm Res*
    **23**(9):2036–2049.

54. Comets E, Brendel K, Mentré F (2008). Computing normalised prediction
    distribution errors to evaluate nonlinear mixed-effect models: the npde
    add-on package for R. *Comput Methods Programs Biomed*
    **90**(2):154–166.

### Stochastic simulation and re-estimation (SSE)

55. Holford NHG, Kimko HC, Monteleone JPR, Peck CC (2000). Simulation of
    clinical trials. *Annu Rev Pharmacol Toxicol* **40**:209–234.

---

## Optimal design

### Population Fisher information matrix (PFIM)

56. Mentré F, Mallet A, Baccar D (1997). Optimal design in random-effects
    regression models. *Biometrika* **84**(2):429–442.

57. Dumont C, Lestini G, Le Nagard H, Mentré F, Comets E, Foulon N,
    Group PD (2014). PFIM 4.0, an extended R program for design evaluation
    and optimisation in nonlinear mixed-effect models.
    *Comput Methods Programs Biomed* **116**(3):234–246.

---

## Prior distributions (MAP / NWPRI)

58. Gelman A, Carlin JB, Stern HS, Dunson DB, Vehtari A, Rubin DB (2013).
    *Bayesian Data Analysis*, 3rd ed. CRC Press, Boca Raton.
    *(Chapter 13 covers MAP estimation and Gaussian priors in hierarchical
    models.)*

59. Wade JR, Beal SL, Sambol NC (1994). Interaction between structural, statistical,
    and covariate models in population pharmacokinetic analysis.
    *J Pharmacokinet Biopharm* **22**(2):165–177.

---

## Software and numerical tools

60. Harris CR, Millman KJ, van der Walt SJ, et al. (2020). Array programming
    with NumPy. *Nature* **585**(7825):357–362.

61. Virtanen P, Gommers R, Oliphant TE, et al. (2020). SciPy 1.0:
    fundamental algorithms for scientific computing in Python.
    *Nat Methods* **17**(3):261–272.

62. McKinney W (2010). Data structures for statistical computing in Python.
    *Proc 9th Python Sci Conf*, pp 56–61. *(pandas.)*

63. Byrd RH, Lu P, Nocedal J, Zhu C (1995). A limited memory algorithm for
    bound constrained optimization. *SIAM J Sci Comput* **16**(5):1190–1208.
    *(L-BFGS-B, the outer optimizer in FO/FOCE/Laplacian.)*

---

## Validation datasets

This section cites the original sources for every dataset used in
`tests/external_validation/`. Dataset provenance is also recorded in the
`dataset_citation` field of each `tests/external_validation/reference/*.json`
file for machine-readable access.

### Theophylline (oral, 12 subjects)

**File:** `tests/external_validation/data/theophylline_boeckmann.csv`
**Used in:** FOCE benchmarks vs. nlmixr2 and Monolix; NCA benchmarks vs.
PKNCA and Phoenix WinNonlin.

64. Boeckmann AJ, Sheiner LB, Beal SL (1992). *NONMEM Users Guide — Part V:
    Introductory Guide*. University of California, San Francisco.
    *(12 subjects, single 320 mg oral dose, plasma theophylline, 132 observations.
    The canonical pharmacometrics learning dataset.)*

65. Pinheiro JC, Bates DM (2000). *Mixed-Effects Models in S and S-PLUS*.
    Springer, New York. *(same data distributed as `nlme::Theoph` and
    `datasets::Theoph` in R; Chapter 6 uses it for one-compartment FOCE
    fitting.)*

### Indomethacin (IV bolus, 6 subjects)

**File:** `tests/external_validation/data/indometh.csv`
**Used in:** NCA benchmarks vs. WinNonlin-backed NonCompart reference tables
(Han 2018).

66. Kwan KC, Breault GO, Umbenhauer ER, McMahon FG, Duggan DE (1976).
    Kinetics of indomethacin absorption, elimination, and enterohepatic
    circulation in man. *J Pharmacokinet Biopharm* **4**(3):255–280.
    *(6 subjects, 25 mg IV bolus, 11 plasma samples each; original clinical
    study underlying the `datasets::Indometh` R dataset.)*

### Warfarin PK (oral, ~32 subjects)

**Files:** `tests/external_validation/data/warfarin_pk.csv`,
`warfarin_pkpd.csv`, `warfarin_pkpd_4.csv`, `warfarin_pkpd_6.csv`
**Used in:** FOCE benchmarks vs. nlmixr2 warfarin fits.

67. Holford NHG (1986). Clinical pharmacokinetics and pharmacodynamics of
    warfarin: understanding the dose-effect relationship.
    *Clin Pharmacokinet* **11**(6):483–504.
    *(foundational warfarin PK/PD characterisation; the dataset form used
    in nlmixr2 tutorials is derived from this work.)*

68. Schoemaker R, Fidler M, Laveille C, et al. (2019). nlmixr: an R package
    for nonlinear mixed-effects model building and diagnostics.
    *CPT Pharmacometrics Syst Pharmacol* **8**(9):641–654.
    *(warfarin PK/PD dataset as distributed with the nlmixr2 package.)*

### Phenobarbital neonatal (simulated, 59 subjects)

**File:** `tests/external_validation/data/phenobarbital_simulated.csv`
**Used in:** FO benchmark vs. published Grasela & Donn (1985) population
estimates.

69. Grasela TH Jr, Donn SM (1985). Neonatal population pharmacokinetics of
    phenobarbital derived from routine clinical data. *Dev Pharmacol Ther*
    **8**(6):374–383.
    *(59 preterm neonates, repeated IV dosing, weight-normalised CL/V.
    The OpenPKPD dataset is simulated from the published parameters
    [CL=0.0047 L/h/kg, V=0.96 L/kg] with seed=42 to allow reproducible
    unit testing.)*

### NONMEM tutorial datasets (Bauer 2019)

**Files:** reference values in `nonmem_402_focei.json`, `nonmem_504_focei.json`,
`nonmem_504f_focei.json`.
**Used in:** FOCE benchmarks vs. NONMEM 7.4.3/7.5.0.

70. Bauer RJ (2019). NONMEM tutorial part II: estimation methods and
    diagnostics for nonlinear mixed-effects models with examples from
    pharmacokinetic and pharmacodynamic studies.
    *CPT Pharmacometrics Syst Pharmacol* **8**(8):538–556. PMC6709422.
    *(Dataset 402: 30 subjects, IV bolus, 2-compartment PK, run with
    ADVAN3 TRANS4. Dataset 504: 60 subjects, IV infusion, 1-compartment
    with WT/AGE/SEX power-law covariates, run with ADVAN1 TRANS2.)*

---

## Validation reference outputs

The following tools and publications provide the reference parameter values
and NCA outputs that the `tests/external_validation/` suite compares against:

71. Han S (2018). Validation of noncompartmental analysis performed by
    NonCompart and R for the estimation of pharmacokinetic parameters.
    *Transl Clin Pharmacol* **26**(1):10–17.
    *(Appendix A republishes WinNonlin-backed Indometh NCA tables used as
    reference in `test_vs_winnonlin_indometh.py`.)*

72. Comets E, Lavenu A, Lavielle M (2017). Parameter estimation in nonlinear
    mixed effect models using saemix, an R implementation of the SAEM
    algorithm. *J Stat Softw* **80**(3):1–41.
    *(Monolix SAEM reference parameters for theophylline, obtained via the
    `monolix2rx` R package conversion vignette.)*

73. Nugent R, et al. (2023). PKNCA: an R package for non-compartmental
    analysis of pharmacokinetic data. *J Pharmacokinet Pharmacodyn*.
    *(theophylline NCA reference values taken from the published PKNCA
    vignette "Computing NCA Parameters for Theophylline".)*

---

*For a summary of how OpenPKPD compares numerically against NONMEM, nlmixr2,
Monolix, and WinNonlin on these datasets, see
`docs/user_guide/external_validation_benchmarks.md`.*
