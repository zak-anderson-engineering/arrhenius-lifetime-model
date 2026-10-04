import numpy as np
import matplotlib.pyplot as plt
from generate_data import make_dataset, EA_TRUE, rate_constant
from arrhenius_fit import fit_rates, arrhenius_fit, lifetime, SERVICE_T_C

if __name__ == "__main__":
    rng = np.random.default_rng(seed=100)
    n_datasets = 500

    Eas, lives = [], []
    for _ in range(n_datasets):
        df = make_dataset(rng)                 # a brand new noisy experiment
        temps, ks, _ = fit_rates(df)
        Ea, A = arrhenius_fit(temps, ks)
        Eas.append(Ea / 1000)
        lives.append(lifetime(Ea, A, SERVICE_T_C) / 8766)

    Eas = np.array(Eas)
    lives = np.array(lives)
    true_Ea = EA_TRUE / 1000
    true_life = np.log(2) / rate_constant(SERVICE_T_C + 273.15) / 8766

    print(f"Ea:       mean {Eas.mean():.2f} kJ/mol, std {Eas.std():.2f}  (true {true_Ea:.0f}, "
          f"bias {100 * (Eas.mean() - true_Ea) / true_Ea:+.2f}%)")
    print(f"Lifetime: mean {lives.mean():.1f} years, std {lives.std():.1f}  (true {true_life:.1f})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.hist(Eas, bins=40, alpha=0.7)
    ax1.axvline(true_Ea, color="k", label="True value")
    ax1.axvline(Eas.mean(), color="r", linestyle="--", label="Mean of fits")
    ax1.set_xlabel("Fitted activation energy (kJ/mol)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Parameter recovery: {n_datasets} synthetic datasets")
    ax1.legend()

    ax2.hist(lives, bins=40, alpha=0.7)
    ax2.axvline(true_life, color="k", label="True value")
    ax2.axvline(lives.mean(), color="r", linestyle="--", label="Mean of fits")
    ax2.set_xlabel(f"Predicted lifetime at {SERVICE_T_C} °C (years)")
    ax2.set_title("Spread of lifetime predictions")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("figures/parameter_recovery.png", dpi=150)
    plt.show()