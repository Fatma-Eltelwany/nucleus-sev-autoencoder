import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

def find_linear_range(dh, ch=0, n_bins=100, 
                      tolerance=0.1,
                      ph_quantile_lo=0.01,
                      ph_quantile_hi=0.99):
    """
    Determines the linear range of the TES response by finding where
    rise_time and decay_time are constant as a function of pulse_height.

    Parameters
    ----------
    dh : cait DataHandler
    ch : int
    n_bins : int
        Number of pulse height bins for the running statistics.
    tolerance : float
        Maximum fractional deviation from the median rise/decay time
        to be considered 'constant' (default 0.1 = 10%).
    ph_quantile_lo, ph_quantile_hi : float
        Quantile range to consider (clips extreme outliers).

    Returns
    -------
    linear_ranges : list of tuple
        List of (ph_lo, ph_hi) tuples defining linear regions.
    """

    ph  = dh.get('events', 'pulse_height', ch).ravel()
    rt  = dh.get('events', 'rise_time',    ch).ravel()
    dt  = dh.get('events', 'decay_time',   ch).ravel()

    # --- obvious garbage ---
    valid = (np.isfinite(ph) & np.isfinite(rt) & np.isfinite(dt)
             & (ph > 0) & (rt > 0) & (dt > 0))
    ph, rt, dt = ph[valid], rt[valid], dt[valid]

    # --- clip to quantile range ---
    ph_lo_clip = np.quantile(ph, ph_quantile_lo)
    ph_hi_clip = np.quantile(ph, ph_quantile_hi)
    mask = (ph >= ph_lo_clip) & (ph <= ph_hi_clip)
    ph, rt, dt = ph[mask], rt[mask], dt[mask]

    # --- binned median of rise and decay time vs pulse height ---
    bins = np.linspace(ph.min(), ph.max(), n_bins + 1)

    rt_median, edges, _ = binned_statistic(ph, rt, statistic='median', bins=bins)
    dt_median, _,     _ = binned_statistic(ph, dt, statistic='median', bins=bins)
    rt_std,    _,     _ = binned_statistic(ph, rt, statistic='std',    bins=bins)
    dt_std,    _,     _ = binned_statistic(ph, dt, statistic='std',    bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # --- find flat region ---
    # use median of the middle 50% of PH range as reference
    # (avoids noise floor and saturation contaminating the reference)
    mid_mask = ((centres > np.quantile(centres, 0.25)) &
                (centres < np.quantile(centres, 0.75)))

    rt_ref = np.nanmedian(rt_median[mid_mask])
    dt_ref = np.nanmedian(dt_median[mid_mask])

    rt_flat = np.abs(rt_median - rt_ref) / (rt_ref + 1e-8) < tolerance
    dt_flat = np.abs(dt_median - dt_ref) / (dt_ref + 1e-8) < tolerance
    both_flat = rt_flat & dt_flat & np.isfinite(rt_median) & np.isfinite(dt_median)

    if both_flat.sum() == 0:
        raise RuntimeError("No flat region found. Try increasing tolerance.")

    flat_centres = centres[both_flat]
    ph_lo = flat_centres.min()
    ph_hi = flat_centres.max()

    print(f"Linear range: [{ph_lo:.4f}, {ph_hi:.4f}] ADU/V")
    print(f"  Rise time ref  : {rt_ref:.4f}  "
          f"(tolerance ±{tolerance*100:.0f}%)")
    print(f"  Decay time ref : {dt_ref:.4f}  "
          f"(tolerance ±{tolerance*100:.0f}%)")

    # --- plot ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for ax, median, std, ref, label in [
        (axes[0], rt_median, rt_std, rt_ref, 'Rise time'),
        (axes[1], dt_median, dt_std, dt_ref, 'Decay time'),
    ]:
        ax.plot(centres, median, color='steelblue', linewidth=1.5)
        ax.fill_between(centres,
                        median - std, median + std,
                        alpha=0.2, color='steelblue', label='±1 std')
        ax.axhline(ref * (1 + tolerance), color='crimson',
                   linestyle='--', linewidth=1, label=f'±{tolerance*100:.0f}% tolerance')
        ax.axhline(ref * (1 - tolerance), color='crimson',
                   linestyle='--', linewidth=1)
        ax.axhline(ref, color='crimson', linestyle='-',
                   linewidth=1, alpha=0.5, label='reference')
        ax.axvspan(ph_lo, ph_hi, alpha=0.12, color='green',
                   label='Linear range')
        ax.axvline(ph_lo, color='green', linestyle=':', linewidth=1.5)
        ax.axvline(ph_hi, color='green', linestyle=':', linewidth=1.5)
        ax.set_ylabel(label)
        ax.legend(fontsize=8, loc='upper right')

    axes[1].set_xlabel('Pulse height [ADU/V]')
    axes[0].set_title(f'Linear range determination — channel {ch}')
    plt.tight_layout()
    plt.show()

    return [(ph_lo, ph_hi)]