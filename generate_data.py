import numpy as np
import pandas as pd
from pathlib import Path

# "True" parameters used to make the synthetic data.
# The fitting code should be able to recover these.
EA_TRUE = 100e3         # activation energy, J/mol
R = 8.314               # gas constant, J/(mol K)
T_REF = 150 + 273.15    # reference temperature, K
K_REF = 1.4e-3          # degradation rate at T_REF, per hour
NOISE = 0.03            # 3% measurement noise


def rate_constant(T_kelvin):
    """Arrhenius law, written relative to a reference temperature."""
    return K_REF * np.exp(-EA_TRUE / R * (1 / T_kelvin - 1 / T_REF))


if __name__ == "__main__":
    rng = np.random.default_rng(seed=3)
    temps_C = [110, 120, 135, 150]     # accelerated ageing temperatures
    rows = []

    for T_C in temps_C:
        k = rate_constant(T_C + 273.15)
        t_eol = np.log(2) / k                      # time to drop to 50% (end of life)
        times = np.linspace(0, 1.5 * t_eol, 8)     # 8 measurements per temperature
        retention = np.exp(-k * times)             # fraction of original property left
        measured = retention * (1 + NOISE * rng.standard_normal(times.size))
        for t, r in zip(times, measured):
            rows.append({"temperature_C": T_C, "time_h": round(t, 1), "retention": round(r, 4)})

    Path("data").mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("data/ageing_data.csv", index=False)
    print(df)