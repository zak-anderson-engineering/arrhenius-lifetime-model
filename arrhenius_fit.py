import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from generate_data import EA_TRUE, R

SERVICE_T_C = 70     # temperature in service, where we want the lifetime
EOL = 0.5            # end of life = 50% retention


def decay(t, k):
    """Retention falls exponentially at rate k."""
    return np.exp(-k * t)


def fit_rates(df):
    """Fit the degradation rate k at each temperature (non-linear least squares)."""
    temps, ks, k_errs = [], [], []
    for T_C, group in df.groupby("temperature_C"):
        popt, pcov = curve_fit(decay, group["time_h"], group["retention"], p0=[1e-3])
        temps.append(T_C)
        ks.append(popt[0])
        k_errs.append(np.sqrt(pcov[0, 0]))
    return np.array(temps), np.array(ks), np.array(k_errs)


def arrhenius_fit(temps_C, ks):
    """Straight-line fit of ln k against 1/T. The slope is -Ea/R."""
    inv_T = 1 / (temps_C + 273.15)
    slope, intercept = np.polyfit(inv_T, np.log(ks), 1)
    return -slope * R, np.exp(intercept)      # Ea (J/mol), A (per hour)


def lifetime(Ea, A, T_C):
    """Time to reach end of life at temperature T_C, in hours."""
    k = A * np.exp(-Ea / (R * (T_C + 273.15)))
    return -np.log(EOL) / k


if __name__ == "__main__":
    df = pd.read_csv("data/ageing_data.csv")
    temps, ks, k_errs = fit_rates(df)
    Ea, A = arrhenius_fit(temps, ks)
    t_life = lifetime(Ea, A, SERVICE_T_C)

    print("Fitted degradation rates:")
    for T, k, e in zip(temps, ks, k_errs):
        print(f"  {T} C: k = {k:.3e} +/- {e:.1e} per hour")
    print(f"\nActivation energy: {Ea/1000:.1f} kJ/mol (true value {EA_TRUE/1000:.0f} kJ/mol)")
    print(f"Predicted lifetime at {SERVICE_T_C} C: {t_life:,.0f} hours = {t_life/8766:.1f} years")

    Path("figures").mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: data and fitted decay curves at each temperature
    for T, k in zip(temps, ks):
        group = df[df["temperature_C"] == T]
        t_smooth = np.linspace(0, group["time_h"].max(), 100)
        line, = ax1.plot(t_smooth, decay(t_smooth, k), label=f"{T} °C")
        ax1.plot(group["time_h"], group["retention"], "o", color=line.get_color())
    ax1.axhline(EOL, color="grey", linestyle="--", label="End of life (50%)")
    ax1.set_xlabel("Ageing time (h)")
    ax1.set_ylabel("Retention (fraction of original)")
    ax1.set_title("Accelerated ageing: data and exponential fits")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Arrhenius plot, extrapolated down to service temperature
    inv_T = 1000 / (temps + 273.15)
    inv_service = 1000 / (SERVICE_T_C + 273.15)
    x_line = np.linspace(inv_T.min(), inv_service, 100)
    ax2.errorbar(inv_T, np.log(ks), yerr=k_errs / ks, fmt="o", label="Fitted rates")
    ax2.plot(x_line, np.log(A) - Ea / (R * 1000) * x_line, "--", label=f"Arrhenius fit, Ea = {Ea/1000:.1f} kJ/mol")
    ax2.plot(inv_service, np.log(A) - Ea / (R * 1000) * inv_service, "s", markersize=8,
             label=f"Extrapolated to {SERVICE_T_C} °C")
    ax2.set_xlabel("1000 / T (K$^{-1}$)")
    ax2.set_ylabel("ln k  (k in h$^{-1}$)")
    ax2.set_title("Arrhenius plot")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("figures/arrhenius_fit.png", dpi=150)
    plt.show()