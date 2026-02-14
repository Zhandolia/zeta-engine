#!/usr/bin/env python3
"""
ζ-Field Analysis — SanDisk (SNDK)
===================================
Fetches real historical SanDisk price data and runs the ζ-field
composite model to generate trading signals and a backtest.

SanDisk traded as SNDK until its acquisition by Western Digital
in May 2016. We analyze available historical data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import sys
import os

# Import our ζ-field model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeta_field_python import xi_field, zeta_signal, run_backtest

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. FETCH REAL DATA
# ─────────────────────────────────────────────

def fetch_sandisk_data():
    """Fetch SanDisk (SNDK) historical data via yfinance."""
    import yfinance as yf

    print("[*] Fetching SanDisk (SNDK) data from Yahoo Finance …")

    # SNDK was delisted in 2016 after WDC acquisition
    # Try fetching; if unavailable, fall back to WDC post-acquisition data
    ticker = yf.Ticker("SNDK")
    hist = ticker.history(period="max")

    if hist.empty:
        print("[!] SNDK data unavailable (delisted). Trying WDC (Western Digital) …")
        ticker = yf.Ticker("WDC")
        hist = ticker.history(start="2006-01-01", end="2024-12-31")
        symbol = "WDC (Western Digital, acquired SanDisk)"
    else:
        symbol = "SNDK (SanDisk)"

    if hist.empty:
        print("[✗] Could not fetch data. Check your internet connection.")
        sys.exit(1)

    prices = hist["Close"].values
    dates = hist.index.to_pydatetime()

    print(f"[✓] Fetched {len(prices)} trading days for {symbol}")
    print(f"    Date range: {dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}")
    print(f"    Price range: ${prices.min():.2f} → ${prices.max():.2f}")

    return prices, dates, symbol


# ─────────────────────────────────────────────
# 2. ζ-FIELD SURFACE FOR SANDISK RETURNS
# ─────────────────────────────────────────────

def plot_sandisk_field(prices, dates, symbol, save_path="sandisk_zeta_field.png"):
    """Plot the ζ-field surface using SanDisk's actual return distribution."""
    from matplotlib.colors import LinearSegmentedColormap

    rets = np.diff(np.log(prices)) * 100.0  # log returns scaled

    # Build phase-space from consecutive returns
    phi_t = rets[1:]     # Φ(t)
    phi_t1 = rets[:-1]   # Φ(t-1)

    # Create grid centered on the actual return distribution
    ret_std = np.std(rets)
    bound = max(4.0 * ret_std, 5.0)
    N = 200
    lin = np.linspace(-bound, bound, N)
    X, Y = np.meshgrid(lin, lin)

    # Compute median time step from data
    median_t = len(prices) / 2.0
    Z = xi_field(X, Y, t=median_t, xi_prev=50.0)

    # Custom colormap
    colors_list = [
        (0.00, "#050520"), (0.20, "#0a1a5c"), (0.40, "#1e6fa0"),
        (0.55, "#5ec8e0"), (0.70, "#b8eaf0"), (0.85, "#f5f7a8"),
        (0.95, "#f0a030"), (1.00, "#e03010"),
    ]
    cmap = LinearSegmentedColormap.from_list(
        "zeta", [(v, c) for v, c in colors_list]
    )

    fig = plt.figure(figsize=(14, 10), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")

    surf = ax.plot_surface(
        X, Y, Z, cmap=cmap, edgecolor="none",
        alpha=0.92, rstride=2, cstride=2, antialiased=True,
    )

    # Scatter actual return pairs on the surface
    # Sample to avoid overcrowding
    sample_n = min(500, len(phi_t))
    idx = np.random.default_rng(42).choice(len(phi_t), sample_n, replace=False)
    sample_x = phi_t[idx]
    sample_y = phi_t1[idx]
    sample_z = xi_field(sample_x, sample_y, t=median_t, xi_prev=50.0)

    ax.scatter(sample_x, sample_y, sample_z + 2, c="white", s=1, alpha=0.4)

    ax.set_xlabel("Φ(t) — Return Today (%)", color="white", fontsize=11, labelpad=12)
    ax.set_ylabel("Φ(t-1) — Return Yesterday (%)", color="white", fontsize=11, labelpad=12)
    ax.set_zlabel("Φ DENSITY", color="white", fontsize=11, labelpad=10)
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("gray")
    ax.yaxis.pane.set_edgecolor("gray")
    ax.zaxis.pane.set_edgecolor("gray")
    ax.view_init(elev=30, azim=-60)

    fig.suptitle(
        f"ζ-Field Analysis — {symbol}\n"
        f"Ξₜ(x,y) = Norm₍₀,₁₀₀₎[V+F+C+B+S+A+H+J+Q+T]",
        color="white", fontsize=14, fontweight="bold", y=0.95,
    )

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=18, pad=0.1)
    cbar.set_label("PERIODICITY", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor="black", bbox_inches="tight")
    print(f"[✓] SanDisk ζ-field surface saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 3. BACKTEST WITH REAL DATA
# ─────────────────────────────────────────────

def plot_sandisk_backtest(prices, dates, symbol,
                          save_path="sandisk_zeta_backtest.png"):
    """Backtest ζ-field signals on real SanDisk data."""
    n = len(prices)

    # Buy & Hold
    bh_equity = 100.0 * prices / prices[0]

    # ZETA_σ.3
    print("[*] Computing ZETA_σ.3 signals …")
    sig_z = zeta_signal(prices, lookback=60, sigma_threshold=0.3)
    eq_z = run_backtest(prices, sig_z)

    # ECLIPSE_σ.4
    print("[*] Computing ECLIPSE_σ.4 signals …")
    sig_e = zeta_signal(prices, lookback=60, sigma_threshold=0.4)
    eq_e = run_backtest(prices, sig_e)

    # Stats
    def max_dd(eq):
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        return dd.max() * 100.0

    def sharpe(eq, rf=0.02):
        rets = np.diff(eq) / eq[:-1]
        excess = rets - rf / 252.0
        if np.std(excess) < 1e-12:
            return 0.0
        return np.sqrt(252) * np.mean(excess) / np.std(excess)

    def cagr(eq, n_days):
        if eq[-1] <= 0 or eq[0] <= 0:
            return 0.0
        years = n_days / 252.0
        return (((eq[-1] / eq[0]) ** (1.0 / years)) - 1.0) * 100.0

    # Count active signals
    n_long = np.sum(sig_z == 1)
    n_short = np.sum(sig_z == -1)
    n_flat = np.sum(sig_z == 0)

    bh_final = float(bh_equity[-1])
    z_final = float(eq_z[-1])
    e_final = float(eq_e[-1])

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor="black",
                              gridspec_kw={"height_ratios": [3, 1]})

    # ── Top panel: equity curves ──
    ax = axes[0]
    ax.set_facecolor("black")

    ax.semilogy(dates, bh_equity, color="#aaaaaa", linewidth=1.2, linestyle="--",
                label=f"BUY_&_HOLD  ${bh_final:.0f}  CAGR={cagr(bh_equity, n):.1f}%")
    ax.semilogy(dates, eq_z, color="#6a5acd", linewidth=1.5,
                label=f"ZETA_σ.3  ${z_final:.0f}  CAGR={cagr(eq_z, n):.1f}%")
    ax.semilogy(dates, eq_e, color="#00bfff", linewidth=1.5,
                label=f"ECLIPSE_σ.4  ${e_final:.0f}  CAGR={cagr(eq_e, n):.1f}%")

    # Drawdown shading for BH
    peak = np.maximum.accumulate(bh_equity)
    dd = (peak - bh_equity) / peak
    dd_mask = dd > 0.10
    start = None
    for i in range(1, n):
        if dd_mask[i] and not dd_mask[i - 1]:
            start = dates[i]
        if not dd_mask[i] and dd_mask[i - 1] and start is not None:
            ax.axvspan(start, dates[i], color="red", alpha=0.12)

    ax.set_ylabel("PORTFOLIO_VALUE ($)", color="white", fontsize=12)
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", fontsize=10, facecolor="#111111",
              edgecolor="gray", labelcolor="white")
    ax.set_title(
        f"ζ-Field Backtest — {symbol}",
        color="cyan", fontsize=14, fontweight="bold",
    )
    ax.grid(color="#222222", linewidth=0.3)

    # ── Bottom panel: signal regimes ──
    ax2 = axes[1]
    ax2.set_facecolor("black")
    colors = np.where(sig_z == 1, "#00cc66",
             np.where(sig_z == -1, "#ff4444", "#333333"))
    ax2.bar(dates, sig_z, width=2, color=colors, alpha=0.8)
    ax2.set_ylabel("SIGNAL", color="white", fontsize=11)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(["SHORT", "FLAT", "LONG"], fontsize=9)
    ax2.tick_params(colors="white", labelsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.grid(color="#222222", linewidth=0.3)

    # Footer
    footer = (
        f"MAX_DD — BH: {max_dd(bh_equity):.1f}%  |  "
        f"ZETA: {max_dd(eq_z):.1f}%  |  ECLIPSE: {max_dd(eq_e):.1f}%     "
        f"SHARPE — BH: {sharpe(bh_equity):.2f}  |  "
        f"ZETA: {sharpe(eq_z):.2f}  |  ECLIPSE: {sharpe(eq_e):.2f}     "
        f"SIGNALS — Long: {n_long}  Short: {n_short}  Flat: {n_flat}"
    )
    fig.text(0.5, 0.01, footer, ha="center", color="gray", fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)
    plt.savefig(save_path, dpi=200, facecolor="black", bbox_inches="tight")
    print(f"[✓] SanDisk backtest saved → {save_path}")
    plt.close()

    # Print summary
    print()
    print(f"  {'Strategy':<18} {'Final':>8} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8}")
    print(f"  {'─' * 50}")
    print(f"  {'BUY_&_HOLD':<18} ${bh_final:>6.0f} {cagr(bh_equity, n):>7.1f}% {max_dd(bh_equity):>7.1f}% {sharpe(bh_equity):>7.2f}")
    print(f"  {'ZETA_σ.3':<18} ${z_final:>6.0f} {cagr(eq_z, n):>7.1f}% {max_dd(eq_z):>7.1f}% {sharpe(eq_z):>7.2f}")
    print(f"  {'ECLIPSE_σ.4':<18} ${e_final:>6.0f} {cagr(eq_e, n):>7.1f}% {max_dd(eq_e):>7.1f}% {sharpe(eq_e):>7.2f}")


# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ζ-Field Analysis — SanDisk")
    print("=" * 60)
    print()

    prices, dates, symbol = fetch_sandisk_data()
    print()

    print("[1/2] Generating ζ-Field surface for SanDisk returns …")
    plot_sandisk_field(prices, dates, symbol, "sandisk_zeta_field.png")
    print()

    print("[2/2] Running ζ-Field backtest on SanDisk …")
    plot_sandisk_backtest(prices, dates, symbol, "sandisk_zeta_backtest.png")

    print()
    print("Done. Output files:")
    print("  • sandisk_zeta_field.png     — 3D field surface with real return scatter")
    print("  • sandisk_zeta_backtest.png  — backtest with equity curves + signals")
