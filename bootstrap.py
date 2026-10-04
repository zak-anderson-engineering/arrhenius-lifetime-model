import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arrhenius_fit import fit_rates, arrhenius_fit, lifetime, SERVICE_T_C
from generate_data import EA_TRUE, rate_constant

if __name__ == "__main__":
    rng = np.random.default_rng(seed=11)
    df = pd.read_csv("data/ageing_data.csv")
    n_boot = 2000

    Ea_samples, life_samples = [], []
    for _ in range(n_boot):
        # Resample the measurements at each temperature, with replacement
        parts = []
        for T_C, g in df.groupby("temperature_C"):
            idx = rng.integers(0, len(g), len(g))
            parts.append(g.iloc[idx])
        resampled = pd.concat(parts)

        temps, ks, _ = fit_rates(resampled)
        Ea, A = arrhenius_fit(temps, ks)
        Ea_samples.append(Ea / 1000)
        life_samples.append(lifetime(Ea, A, SERVICE_T_C) / 8766)   # years

    Ea_samples = np.array(Ea_samples)
    life_samples = np.array(life_samples)
    true_life = np.log(2) / rate_constant(SERVICE_T_C + 273.15) / 8766

    Ea_lo, Ea_hi = np.percentile(Ea_samples, [2.5, 97.5])
    life_lo, life_hi = np.percentile(life_samples, [2.5, 97.5])

    print(f"Ea: median {np.median(Ea_samples):.1f} kJ/mol, 95% CI [{Ea_lo:.1f}, {Ea_hi:.1f}]  (true {EA_TRUE/1000:.0f})")
    print(f"Lifetime at {SERVICE_T_C} C: median {np.median(life_samples):.1f} years, "
          f"95% CI [{life_lo:.1f}, {life_hi:.1f}]  (true {true_life:.1f})")
    print("True Ea inside CI:", Ea_lo <= EA_TRUE / 1000 <= Ea_hi)
    print("True lifetime inside CI:", life_lo <= true_life <= life_hi)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.hist(Ea_samples, bins=50, alpha=0.7)
    ax1.axvline(EA_TRUE / 1000, color="k", label="True value")
    ax1.axvline(Ea_lo, color="r", linestyle="--", label="95% CI")
    ax1.axvline(Ea_hi, color="r", linestyle="--")
    ax1.set_xlabel("Activation energy (kJ/mol)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Bootstrap: activation energy ({n_boot} resamples)")
    ax1.legend()

    ax2.hist(life_samples, bins=50, alpha=0.7)
    ax2.axvline(true_life, color="k", label="True value")
    ax2.axvline(life_lo, color="r", linestyle="--", label="95% CI")
    ax2.axvline(life_hi, color="r", linestyle="--")
    ax2.set_xlabel(f"Predicted lifetime at {SERVICE_T_C} °C (years)")
    ax2.set_title("Bootstrap: predicted lifetime")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("figures/bootstrap.png", dpi=150)
    plt.show()