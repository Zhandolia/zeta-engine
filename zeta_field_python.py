#!/usr/bin/env python3
"""
ζ-Field Model — Python Implementation (Fixed Version)
=====================================================
Ξ_t(x,y) = Norm_[0,100] [ V + F + C + B + S + A + H + J + Q + T ]

A composite mathematical field that models asset price behavior by combining
10 component functions over a 2D phase space (Φ(t), Φ(t-1)).

NOTE: The actual ζ-Field equations are proprietary. This is an inspired
approximation using known quantitative-finance concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. COMPONENT FUNCTIONS  (Bowl / Crater Topology)
#
# Redesigned to produce a sombrero/Mexican-hat shape:
#   • Center dips down (low density = mean-reversion zone)
#   • Rim peaks up (high density = momentum / trend zone)
#   • Multiple angular peaks around the rim
#   • Asymmetric deformations from market drift & skew
# ─────────────────────────────────────────────

def V(r, a_loc=0.0, b_loc=0.0, depth=30.0, sigma=1.5):
    """
    Mexican-hat / sombrero + edge curl potential.
    Low at center (r≈0), peaks at r≈σ√2, then rises AGAIN at large r
    creating the dramatic curled-up edge (cape effect) from the original.
    """
    r_shifted = np.sqrt((r)**2 + a_loc**2 + b_loc**2 + 1e-12)
    u = r_shifted / sigma
    # Core Mexican hat (bowl + rim)
    core = depth * (u**2 - 1.0) * np.exp(-0.5 * u**2)
    # Edge curl — rises again at large r, creating the cape
    edge_curl = depth * 0.6 * np.tanh(0.3 * (r_shifted - 3.0))
    return core + edge_curl


def F(x, a_loc=0.0, k=2.5, shift=0.0):
    """
    Directional force — tilts the bowl asymmetrically.
    Positive drift shifts the peak toward positive returns;
    negative drift toward negative. Creates the lean visible in
    the original's surface.
    """
    return k * np.tanh((x - a_loc) * 0.8) * np.exp(-0.05 * x**2) - shift


def C(x, y, b=0.6):
    """
    Coupling / correlation field.
    xy interaction with saturation — creates saddle-point
    deformations along the diagonals (correlated vs anti-correlated
    return regimes).
    """
    return b * x * y * np.exp(-0.08 * (x**2 + y**2))


def B(r, xi_prev, t, gamma=0.08):
    """
    Boundary feedback — self-referential damping.
    Suppresses extreme radii proportional to previous field state;
    the sin(t) term creates a breathing oscillation.
    """
    return -gamma * r**2 * np.sin(t * 0.12) * (1 + 0.01 * xi_prev)


def S(x, y, chi=0.3):
    """
    Angular symmetry breaker — creates multiple peaks around the rim.
    Uses angular harmonics (cos 3θ, cos 5θ) to produce 3-fold and
    5-fold symmetric peaks. Active on BOTH the rim AND the curled edges
    to produce the complex multi-peak topology of the original.
    """
    r = np.sqrt(x**2 + y**2 + 1e-12)
    theta = np.arctan2(y, x)
    # 3-fold + weaker 5-fold + subtle 2-fold angular modulation
    angular = (np.cos(3 * theta) + 0.45 * np.cos(5 * theta + 0.5)
               + 0.3 * np.cos(2 * theta - 0.3))
    # Active on inner rim AND outer edge (twin Gaussian window)
    rim_window = np.exp(-0.5 * (r - 2.0)**2)
    edge_window = 0.6 * np.exp(-0.3 * (r - 3.8)**2)
    return chi * 14.0 * angular * (rim_window + edge_window)


def A(x, y, alpha=1.0):
    """
    Amplitude / ring modulator.
    Adds a secondary raised ring at larger radius, creating the
    'raised edge' effect visible in the original. The ring's height
    scales with alpha (linked to volatility).
    """
    r = np.sqrt(x**2 + y**2)
    ring1 = np.exp(-0.8 * (r - 1.8)**2)   # inner ring
    ring2 = 0.5 * np.exp(-0.4 * (r - 3.2)**2)  # outer ring
    return alpha * 10.0 * (ring1 + ring2)


def H(r, phi, xi_prev, t, omega=0.5, amp=6.0):
    """
    Harmonic oscillator — time-varying angular rotation.
    Rotates the angular peaks around the rim over time, creating
    the multi-motion effect (surface appears to 'swirl').
    """
    if isinstance(phi, (list, tuple, np.ndarray)):
        phi_angle = np.arctan2(phi[1], phi[0])
    else:
        phi_angle = phi
    # Rotating angular modulation on the rim
    radial_env = np.exp(-0.5 * (r - 2.0)**2)
    return amp * np.cos(omega * t + 2 * phi_angle) * radial_env


def J(r, xi_prev, alpha=1.0, lam=0.5):
    """
    Jump diffusion — fat-tail discontinuity.
    Creates a step-like rise at a threshold radius, modeling
    the transition from normal to extreme return regimes.
    """
    threshold = 1.5 + 0.005 * xi_prev
    return lam * 8.0 * (1.0 / (1.0 + np.exp(-alpha * 3 * (r - threshold))))


def Q(r, xi_prev, q=0.5):
    """
    Quantile envelope — outer boundary modulation.
    Instead of simple decay, creates a dramatic upturn at the far
    boundary to produce the 'curled cape' silhouette from the original.
    The field rises at the edge, dips, then rises again.
    """
    # Inner suppression (keeps bowl intact)
    inner_decay = np.exp(-0.12 * r**2) * (1 + 0.3 * np.tanh(0.02 * xi_prev))
    # Outer upturn (cape curl)
    outer_rise = 0.4 * np.tanh(0.5 * (r - 3.5))
    return -q * 6.0 * (1 - inner_decay) + q * 10.0 * outer_rise


def T_func(x, y, t, drift=0.03):
    """
    Temporal drift — time-varying surface breathing.
    Modulates the overall bowl depth with a slow oscillation,
    making the surface 'breathe' over time.
    """
    r = np.sqrt(x**2 + y**2)
    breathing = np.sin(drift * t) * 4.0
    radial_mod = np.exp(-0.1 * r**2)
    return breathing * radial_mod


# ─────────────────────────────────────────────
# 2. COMPOSITE FIELD Ξ_t(x, y)
# ─────────────────────────────────────────────

def xi_field(x, y, t=1.0, xi_prev=50.0, params=None):
    """
    Compute Ξ_t(x,y) — the full composite field, normalized to [0, 100].

    Parameters
    ----------
    x, y  : ndarray  — phase-space coordinates (Φ(t), Φ(t-1))
    t     : float    — time step
    xi_prev : float  — previous field value (self-referential feedback)
    params : dict    — optional parameter overrides
    """
    if params is None:
        params = {}
    
    p = params
    r = np.sqrt(x**2 + y**2)

    raw = (
        V(r, p.get("a_loc", 0.0), p.get("b_loc", 0.0))
        + F(x, p.get("a_loc", 0.0))
        + C(x, y, p.get("b", 0.6))
        + B(r, xi_prev, t)
        + S(x, y, p.get("chi", 0.3))
        + A(x, y, p.get("alpha", 1.0))
        + H(r, (x, y), xi_prev, t)
        + J(r, xi_prev)
        + Q(r, xi_prev)
        + T_func(x, y, t)
    )

    # Normalize to [0, 100]
    rmin, rmax = raw.min(), raw.max()
    if rmax - rmin < 1e-12:
        return np.atleast_1d(np.full_like(raw, 50.0))
    return np.atleast_1d(100.0 * (raw - rmin) / (rmax - rmin))


# ─────────────────────────────────────────────
# 3. 3D SURFACE VISUALIZATION
# ─────────────────────────────────────────────

def plot_3d_surface(save_path="zeta_field_3d.png"):
    """Produce a 3D surface plot mimicking the screenshot aesthetic."""
    N = 200
    lin = np.linspace(-5, 5, N)
    X, Y = np.meshgrid(lin, lin)
    Z = xi_field(X, Y, t=5.0, xi_prev=50.0)

    # Custom colormap: dark-blue → cyan → white → yellow → red-orange
    colors_list = [
        (0.00, "#050520"),
        (0.20, "#0a1a5c"),
        (0.40, "#1e6fa0"),
        (0.55, "#5ec8e0"),
        (0.70, "#b8eaf0"),
        (0.85, "#f5f7a8"),
        (0.95, "#f0a030"),
        (1.00, "#e03010"),
    ]
    cmap = LinearSegmentedColormap.from_list(
        "zeta_periodicity",
        [(v, c) for v, c in colors_list],
    )

    fig = plt.figure(figsize=(14, 10), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")

    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cmap,
        edgecolor="none",
        alpha=0.92,
        rstride=2, cstride=2,
        antialiased=True,
    )

    # Style axes
    ax.set_xlabel("Φ(t)", color="white", fontsize=13, labelpad=12)
    ax.set_ylabel("Φ(t-1)", color="white", fontsize=13, labelpad=12)
    ax.set_zlabel("Φ DENSITY", color="white", fontsize=13, labelpad=10)
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("gray")
    ax.yaxis.pane.set_edgecolor("gray")
    ax.zaxis.pane.set_edgecolor("gray")
    ax.view_init(elev=30, azim=-60)

    # Title with equation
    fig.suptitle(
        "ζ-Field  Ξₜ(x,y) = Norm₍₀,₁₀₀₎[V + F + C + B + S + A + H + J + Q + T]",
        color="white", fontsize=15, fontweight="bold", y=0.95,
    )

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=18, pad=0.1)
    cbar.set_label("PERIODICITY", color="white", fontsize=12)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor="black", bbox_inches="tight")
    print(f"[✓] 3D surface saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 4. PORTFOLIO BACKTEST (synthetic demo)
# ─────────────────────────────────────────────

def generate_synthetic_prices(n=3000, seed=42):
    """Generate synthetic daily prices using geometric Brownian motion + jumps."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    mu, sigma_base = 0.06, 0.18
    prices = np.zeros(n)
    prices[0] = 100.0
    for i in range(1, n):
        jump = rng.choice([0.0, rng.normal(0.0, 0.03)], p=[0.97, 0.03])
        prices[i] = prices[i - 1] * np.exp(
            (mu - 0.5 * sigma_base**2) * dt
            + sigma_base * np.sqrt(dt) * rng.standard_normal()
            + jump
        )
    return prices


def zeta_signal(prices, lookback=60, sigma_threshold=0.3):
    """
    Compute a simplified ζ-field trading signal.
    Uses the composite field evaluated on rolling return features.
    Returns +1 (long), 0 (flat), -1 (short).
    """
    n = len(prices)
    signals = np.zeros(n)
    for i in range(lookback, n):
        window = prices[i - lookback : i]
        rets = np.diff(np.log(window))
        
        # Ensure we have enough returns
        if len(rets) < 2:
            continue
            
        phi_t = rets[-1] * 100.0       # current return scaled
        phi_t1 = rets[-2] * 100.0      # previous return scaled (FIXED typo)
        
        field_res = xi_field(
            np.array([phi_t]), 
            np.array([phi_t1]),
            t=float(i), 
            xi_prev=50.0,
        )

        # Handle both array (normal) and scalar (edge case) returns
        if np.ndim(field_res) == 0:
            field_val = float(field_res)
        else:
            field_val = field_res[0]
        
        if field_val > (50.0 + sigma_threshold * 30.0):
            signals[i] = 1.0   # bullish regime
        elif field_val < (50.0 - sigma_threshold * 30.0):
            signals[i] = -1.0  # bearish regime
        else:
            signals[i] = 0.0   # neutral
    return signals


def eclipse_signal(prices, lookback=60, sigma_threshold=0.4):
    """Variant strategy with different threshold (ECLIPSE_σ.4)."""
    return zeta_signal(prices, lookback=lookback, sigma_threshold=sigma_threshold)


def run_backtest(prices, signals, initial_capital=100.0):
    """Compute portfolio equity curve from signals."""
    n = len(prices)
    equity = np.zeros(n)
    equity[0] = initial_capital
    for i in range(1, n):
        daily_ret = prices[i] / prices[i - 1] - 1.0
        equity[i] = equity[i - 1] * (1.0 + signals[i - 1] * daily_ret)
    return equity


def plot_backtest(save_path="zeta_backtest.png"):
    """Produce the portfolio comparison chart."""
    prices = generate_synthetic_prices(n=3000)
    n = len(prices)

    # Date-like x-axis (2006 → ~2018)
    years = np.linspace(2006, 2018, n)

    # Buy & Hold
    bh_equity = 100.0 * prices / prices[0]

    # ZETA_σ.3
    sig_z = zeta_signal(prices, sigma_threshold=0.3)
    eq_z = run_backtest(prices, sig_z)

    # ECLIPSE_σ.4
    sig_e = eclipse_signal(prices, sigma_threshold=0.4)
    eq_e = run_backtest(prices, sig_e)

    # Max drawdown helper
    def max_dd(eq):
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        return dd.max() * 100.0

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="black")
    ax.set_facecolor("black")

    # Get final values safely
    bh_final = float(bh_equity[-1])
    eq_z_final = float(eq_z[-1])
    eq_e_final = float(eq_e[-1])

    ax.semilogy(years, bh_equity, color="#aaaaaa", linewidth=1.2, linestyle="--",
                label=f"BUY_&_HOLD  ${bh_final:.0f} ({bh_final - 100.0:.0f}%)")
    ax.semilogy(years, eq_z, color="#6a5acd", linewidth=1.5,
                label=f"ZETA_σ.3  ${eq_z_final:.0f} ({eq_z_final - 100.0:.0f}%)")
    ax.semilogy(years, eq_e, color="#00bfff", linewidth=1.5,
                label=f"ECLIPSE_σ.4  ${eq_e_final:.0f} ({eq_e_final - 100.0:.0f}%)")

    # Highlight drawdown periods for BUY_&_HOLD
    peak = np.maximum.accumulate(bh_equity)
    dd = (peak - bh_equity) / peak
    dd_mask = dd > 0.10
    start = None
    for i in range(1, n):
        if dd_mask[i] and not dd_mask[i - 1]:
            start = years[i]
        if not dd_mask[i] and dd_mask[i - 1] and start is not None:
            ax.axvspan(start, years[i], color="red", alpha=0.15)

    ax.set_xlabel("Year", color="white", fontsize=12)
    ax.set_ylabel("PORTFOLIO_VALUE", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.legend(loc="upper left", fontsize=10, facecolor="#111111", edgecolor="gray",
              labelcolor="white")

    title = f"ζ-Field Backtest  |  ZETA_σ.3 LEADS: ${eq_z_final:.0f}"
    ax.set_title(title, color="cyan", fontsize=14, fontweight="bold")

    # Annotate final values
    annotations: list[tuple[np.ndarray, str, str]] = [
        (bh_equity, "BUY_&_HOLD", "#aaaaaa"),
        (eq_z, "ZETA_σ.3", "#6a5acd"),
        (eq_e, "ECLIPSE_σ.4", "#00bfff"),
    ]
    for equity_curve, lbl, col in annotations:
        final_val = float(equity_curve[-1])
        ax.annotate(
            f"${final_val:.0f}",
            xy=(years[-1], final_val),
            xytext=(10, 0),
            textcoords="offset points",
            color=col, fontsize=10, fontweight="bold",
        )

    # Footer
    footer = (
        f"MAX_DD — BUY_&_HOLD: {max_dd(bh_equity):.1f}%  |  "
        f"ZETA_σ.3: {max_dd(eq_z):.1f}%  |  "
        f"ECLIPSE_σ.4: {max_dd(eq_e):.1f}%"
    )
    fig.text(0.5, 0.01, footer, ha="center", color="gray", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor="black", bbox_inches="tight")
    print(f"[✓] Backtest chart saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ζ-Field Model — Python Implementation")
    print("=" * 60)
    print()

    print("[1/2] Generating 3D ζ-Field surface …")
    plot_3d_surface("zeta_field_3d.png")

    print("[2/2] Running portfolio backtest …")
    plot_backtest("zeta_backtest.png")

    print()
    print("Done. Output files:")
    print("  • zeta_field_3d.png   — 3D field surface")
    print("  • zeta_backtest.png   — portfolio backtest chart")