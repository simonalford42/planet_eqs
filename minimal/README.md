# Minimal Predictor

Evaluate on random/resonant split:

```bash
python minimal/2d_plot.py
```

Run the period-ratio grid example from the repo root (requires compatible torch version with the pickled input cache unfortunately):

```bash
python minimal/plot_period_ratio.py
```

Another basic example is in planet_stability.py:
```python
# =====================================================================
# Example: build a 3-planet system in a mildly-resonant configuration
# and predict its log10(T_inst / P_inner).
# =====================================================================
if __name__ == "__main__":
    model = StabilityPredictor.load()
    print(f"Using equation complexity = {model.complexity}")
    print(f"  {model.equation()}")
    print()

    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.ri_whfast.safe_mode = 0
    sim.add(m=1.0)                                  # star (solar mass)
    sim.add(m=1e-4, P=1.0,         theta="uniform")  # planet 1 (P=1)
    sim.add(m=1e-4, P=1.0 / 0.655, theta="uniform")  # planet 2 (P=1.527)
    sim.add(m=1e-4, P=1.0 / 0.655 / 0.655, theta="uniform")  # planet 3
    sim.move_to_com()

    log_t_inst = model.predict(sim)
    print(f"log10(T_inst / P_inner) = {log_t_inst:.4f}")
    print(f"  (means the system is expected to go unstable after about")
    print(f"   10**{log_t_inst:.2f} ~= {10**log_t_inst:.2e} inner-planet periods)")
```

