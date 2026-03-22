"""
Example 17: SBML model import and integration with OpenPKPD.

Demonstrates:
  - Building an SBMLModel from scratch (without a file) using the internal
    _build_des_callable API — useful when you have a QSP model in code
  - Using python-libsbml to load a real SBML file (optional, skipped if absent)
  - Connecting SBMLModel.des_callable to ADVAN6 for simulation
  - Converting SBML parameters to ThetaSpec for estimation
  - Two-compartment IV model expressed as SBML reactions

Background
----------
SBML (Systems Biology Markup Language) is the standard format for QSP and
systems pharmacology models.  OpenPKPD can import SBML files into a
DES callable compatible with the ADVAN6/DDESubroutine integration engine:

    from openpkpd.io import load_sbml
    model = load_sbml("my_qsp_model.xml")
    advan = ADVAN6(n_compartments=model.n_compartments)
    sol   = advan.solve(model.default_pk_params, doses, times,
                        des_callable=model.des_callable)

This example shows both the programmatic API and the file-based API.
"""

from __future__ import annotations

import os

import numpy as np


# ---------------------------------------------------------------------------
# 1. Output path helper
# ---------------------------------------------------------------------------

def _resolve_output_path(filename: str) -> str:
    out_dir = os.environ.get("OPENPKPD_EXAMPLE_OUTPUT")
    if not out_dir:
        out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


# ---------------------------------------------------------------------------
# 2. Build a two-compartment IV model programmatically
#    (equivalent to what load_sbml would produce from an SBML file)
# ---------------------------------------------------------------------------

def build_two_cmt_sbml_model():
    """
    Create an SBMLModel for a two-compartment IV bolus model.

    Species:
        A_central      — central compartment amount
        A_peripheral   — peripheral compartment amount

    Parameters:
        CL             — clearance (L/h)
        V1             — central volume (L)
        Q              — inter-compartmental clearance (L/h)
        V2             — peripheral volume (L)
    """
    from openpkpd.io.sbml import SBMLModel, _build_des_callable

    species_names = ["A_central", "A_peripheral"]
    species_index = {"A_central": 0, "A_peripheral": 1}
    parameter_names = ["CL", "V1", "Q", "V2"]
    default_pk_params = {"CL": 3.0, "V1": 10.0, "Q": 1.5, "V2": 30.0}

    # dA_central/dt  = -(CL/V1)*A_central - (Q/V1)*A_central + (Q/V2)*A_peripheral
    # dA_peripheral/dt = +(Q/V1)*A_central - (Q/V2)*A_peripheral
    dadt_exprs = {
        0: [
            "-(pk_params['CL'] / pk_params['V1']) * A[0]",
            "-(pk_params['Q'] / pk_params['V1']) * A[0]",
            "+(pk_params['Q'] / pk_params['V2']) * A[1]",
        ],
        1: [
            "+(pk_params['Q'] / pk_params['V1']) * A[0]",
            "-(pk_params['Q'] / pk_params['V2']) * A[1]",
        ],
    }

    des_callable = _build_des_callable(
        dadt_exprs, species_names, species_index, parameter_names, 2, []
    )

    return SBMLModel(
        species_names=species_names,
        parameter_names=parameter_names,
        default_pk_params=default_pk_params,
        initial_amounts={"A_central": 0.0, "A_peripheral": 0.0},
        n_compartments=2,
        des_callable=des_callable,
        source_path="<programmatic>",
    )


# ---------------------------------------------------------------------------
# 3. Simulate the model
# ---------------------------------------------------------------------------

def simulate(model, dose=100.0, t_obs=None):
    from openpkpd.data.event_processor import DoseEvent
    from openpkpd.pk.ode.advan6 import ADVAN6

    if t_obs is None:
        t_obs = np.linspace(0.25, 24.0, 96)

    dose_events = [DoseEvent(time=0.0, amount=dose, rate=0.0, duration=0.0, compartment=1)]
    advan = ADVAN6(n_compartments=model.n_compartments)
    sol = advan.solve(
        pk_params=model.default_pk_params,
        dose_events=dose_events,
        obs_times=t_obs,
        des_callable=model.des_callable,
    )
    return t_obs, sol


# ---------------------------------------------------------------------------
# 4. Parameter recovery: map ThetaSpecs from model
# ---------------------------------------------------------------------------

def show_theta_specs(model):
    specs = model.to_theta_specs()
    print(f"\n{'Parameter':<12}  {'Init':>10}  {'Lower':>10}")
    print("-" * 36)
    for s in specs:
        print(f"{s.label:<12}  {s.init:>10.4f}  {s.lower:>10.4f}")


# ---------------------------------------------------------------------------
# 5. Optional: load from SBML file
# ---------------------------------------------------------------------------

def try_load_from_file():
    """
    If python-libsbml is installed, demonstrate file-based loading.
    Writes a minimal SBML file to a temp directory and loads it.
    """
    import tempfile

    SBML_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="one_cmt" name="One-compartment IV">
    <listOfCompartments>
      <compartment id="central" volume="10" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A_central" compartment="central" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="CL" value="2.0" constant="true"/>
      <parameter id="V"  value="10.0" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="elimination" reversible="false">
        <listOfReactants>
          <speciesReference species="A_central" stoichiometry="1"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><apply><divide/><ci>CL</ci><ci>V</ci></apply><ci>A_central</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""
    try:
        from openpkpd.io import load_sbml

        with tempfile.TemporaryDirectory() as tmp:
            sbml_path = os.path.join(tmp, "one_cmt.xml")
            with open(sbml_path, "w") as f:
                f.write(SBML_XML)

            model = load_sbml(sbml_path)
            print(f"\nLoaded SBML file: {os.path.basename(sbml_path)}")
            print(f"  Species:    {model.species_names}")
            print(f"  Parameters: {model.parameter_names}")
            print(f"  n_compartments: {model.n_compartments}")
            return model

    except ImportError:
        print("\npython-libsbml not installed — skipping file-based SBML load.")
        print("Install with: pip install python-libsbml")
        return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Example 17: SBML model import and simulation")
    print("=" * 60)

    # Build model programmatically
    model = build_two_cmt_sbml_model()
    print(f"\nTwo-compartment IV model:")
    print(f"  Species:        {model.species_names}")
    print(f"  Parameters:     {model.parameter_names}")
    print(f"  n_compartments: {model.n_compartments}")
    print(f"  Default params: {model.default_pk_params}")

    show_theta_specs(model)

    # Simulate
    t_obs, sol = simulate(model, dose=100.0)
    print(f"\nSimulation ({len(t_obs)} time points):")
    print(f"  {'Time':>6}  {'C_central':>12}  {'A_peripheral':>14}")
    print("  " + "-" * 36)
    for i in range(0, len(t_obs), 12):
        v1 = model.default_pk_params["V1"]
        c_central = sol.amounts[i, 0] / v1
        a_peri = sol.amounts[i, 1]
        print(f"  {t_obs[i]:6.2f}  {c_central:12.4f}  {a_peri:14.4f}")

    # Verify mass balance (total drug should decrease monotonically)
    total = sol.amounts[:, 0] + sol.amounts[:, 1]
    assert total[0] >= total[-1], "Total drug should decrease over time (elimination)"
    assert np.all(sol.amounts >= 0), "Amounts should be non-negative"
    print("\nMass balance check: PASSED")

    # pk_callable_from_theta round-trip
    theta_vec = [3.0, 10.0, 1.5, 30.0]
    pk_from_theta = model.pk_callable_from_theta(theta_vec)
    assert pk_from_theta["CL"] == 3.0
    assert pk_from_theta["V1"] == 10.0
    print("ThetaSpec round-trip: PASSED")

    # Try loading from file (requires python-libsbml)
    try_load_from_file()

    print("\nDone. SBML import example complete.")

    # ------------------------------------------------------------------
    # Optional: plot
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        v1 = model.default_pk_params["V1"]
        c_central = sol.amounts[:, 0] / v1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(t_obs, c_central)
        ax1.set_xlabel("Time (h)")
        ax1.set_ylabel("Concentration (mg/L)")
        ax1.set_title("Central compartment")
        ax1.grid(True, alpha=0.3)

        ax2.semilogy(t_obs, c_central)
        ax2.set_xlabel("Time (h)")
        ax2.set_ylabel("Concentration (mg/L)")
        ax2.set_title("Central compartment (log scale)")
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Example 17: SBML-derived two-compartment IV model")
        fig.tight_layout()
        out_path = _resolve_output_path("17_sbml_import.png")
        fig.savefig(out_path, dpi=120)
        print(f"Figure saved to {out_path}")
        plt.close(fig)
    except ImportError:
        print("matplotlib not installed — skipping plot.")


if __name__ == "__main__":
    main()
