#!/usr/bin/env python3
"""
Advanced ζ-Field Animation — Institutional Quant Model
=======================================================
Production-grade animated visualization of a 12-component
quantitative surface combining cutting-edge techniques:

  • Heston Stochastic Volatility  
  • Hidden Markov Regime Switching
  • Gaussian Copula Tail Dependencies
  • GARCH(1,1) Conditional Volatility
  • Lévy α-Stable Fat-Tail Processes
  • Variance Risk Premium Surface
  • Merton Jump-Diffusion
  • VaR/CVaR Quantile Risk Mapping
  • Fractional Brownian Motion (Long Memory)
  • Skewness & Kurtosis Higher-Moment Surfaces

AXES: Completely fixed (no camera motion)
OUTPUT: Single GIF with 3D field + portfolio chart
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import warnings
import sys
import os
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeta_field_advanced import xi_field_advanced

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════
N_GRID      = 120
N_FRAMES    = 120
DURATION_S  = 18
LOOKBACK    = 60
DPI         = 110
FIG_W, FIG_H = 14, 16
SAVE_PATH   = "sandisk_zeta_advanced.gif"


# ═══════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════
def fetch_prices():
    import yfinance as yf
    print("[*] Fetching market data …")
    tk = yf.Ticker("SNDK")
    hist = tk.history(period="max")
    if hist.empty:
        print("[!] SNDK → fallback to WDC")
        tk = yf.Ticker("WDC")
        hist = tk.history(start="2006-01-01", end="2024-12-31")
        sym = "WDC"
    else:
        sym = "SNDK"
    prices = hist["Close"].values
    dates  = hist.index.to_pydatetime()
    print(f"[✓] {len(prices)} days for {sym}")
    return prices, dates, sym


# ═══════════════════════════════════════════
# COLORMAP
# ═══════════════════════════════════════════
def advanced_cmap():
    stops = [
        (0.00, "#010312"), (0.07, "#05082c"), (0.14, "#0a0f54"),
        (0.22, "#0f1d7c"), (0.31, "#1a5098"), (0.40, "#2880b0"),
        (0.49, "#38adc8"), (0.58, "#50c0d8"), (0.66, "#70d8d8"),
        (0.73, "#90e8c0"), (0.80, "#b0f090"), (0.86, "#d0f050"),
        (0.91, "#e8d020"), (0.96, "#f89820"), (1.00, "#c01010"),
    ]
    return LinearSegmentedColormap.from_list("advanced", stops)


# ═══════════════════════════════════════════
# REGIME DETECTION (2-state HMM approximation)
# ═══════════════════════════════════════════
def detect_regime(rets, window=30):
    """Simplified HMM: high vol → crisis (1), low vol → calm (0)."""
    if len(rets) < window:
        return 0.0
    vol = np.std(rets[-window:])
    vol_long = np.std(rets)
    # If current vol > 1.5x long-run vol → crisis mode
    return float(vol > 1.5 * vol_long)


# ═══════════════════════════════════════════
# ADVANCED FIELD with full market params
# ═══════════════════════════════════════════
def compute_advanced_field(X, Y, t, rets, vol_prev):
    """Compute 12-component field with market-derived parameters."""
    if len(rets) < 2:
        return np.full_like(X, 50.0)
    
    vol = max(0.1, np.std(rets))
    mean_ret = np.mean(rets)
    
    # Higher moments
    skew = (np.mean(rets**3) / (vol**3 + 1e-12)) if vol > 0.01 else 0.0
    kurt = (np.mean(rets**4) / (vol**4 + 1e-12)) if vol > 0.01 else 3.0
    
    # Regime probability 
    regime_prob = detect_regime(rets)
    
    # Correlation (simplified: autocorr of returns)
    if len(rets) > 10:
        corr_mat = np.corrcoef(rets[:-1], rets[1:])
        rho = float(corr_mat[0, 1]) if not np.isnan(corr_mat[0, 1]) else 0.0
    else:
        rho = 0.0
    
    # Long-run vol (for Heston θ parameter)
    theta_vol = np.std(rets) if len(rets) > 20 else 1.5
    
    params = {
        "vol": vol,
        "vol_prev": vol_prev,
        "mean_ret": mean_ret,
        "skew": skew,
        "kurt": kurt,
        "regime_prob": regime_prob,
        "rho": rho,
        "theta_vol": theta_vol,
    }
    
    return xi_field_advanced(X, Y, t=t, xi_prev=50.0, params=params)


# ═══════════════════════════════════════════
# EQUITY CURVES (pre-computed)
# ═══════════════════════════════════════════
def precompute_equity(prices, rets, n):
    """BH, ZETA, ECLIPSE strategies."""
    bh = 100.0 * prices / prices[0]
    
    sig_z = np.zeros(n)
    sig_e = np.zeros(n)
    for i in range(LOOKBACK, n):
        w = rets[max(0, i - LOOKBACK):i]
        if len(w) < 2:
            continue
        m, s = np.mean(w), np.std(w)
        ratio = m / (s + 1e-12)
        if ratio > 0.03:
            sig_z[i] = 1.0
        elif ratio < -0.03:
            sig_z[i] = -1.0
        if ratio > 0.04:
            sig_e[i] = 1.0
        elif ratio < -0.04:
            sig_e[i] = -1.0
    
    eq_z = np.zeros(n); eq_e = np.zeros(n)
    eq_z[0] = eq_e[0] = 100.0
    for i in range(1, n):
        dr = prices[i] / prices[i - 1] - 1.0
        eq_z[i] = eq_z[i - 1] * (1.0 + sig_z[i - 1] * dr)
        eq_e[i] = eq_e[i - 1] * (1.0 + sig_e[i - 1] * dr)
    
    return bh, eq_z, eq_e


# ═══════════════════════════════════════════
# RENDER FRAME
# ═══════════════════════════════════════════
def render_frame(fi, X, Y, lin, bound, prices, dates, rets, sym,
                 frame_indices, cmap, vol_history, bh, eq_z, eq_e):
    
    idx = frame_indices[fi]
    t = float(idx)
    n = len(prices)
    
    # Rolling window
    w = rets[max(0, idx - LOOKBACK):idx]
    vol         = np.std(w) if len(w) >= 2 else 1.0
    mean_ret    = np.mean(w) if len(w) >= 2 else 0.0
    regime_prob = detect_regime(rets[:idx])
    vol_prev    = vol_history[max(0, fi - 1)]
    
    Z = compute_advanced_field(X, Y, t, w, vol_prev)
    
    # ──────────────────────────────────────
    # FIGURE: 3D field (65%) + chart (35%)
    # ──────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#000508")
    ax3d    = fig.add_axes([0.00, 0.32, 1.0, 0.66], projection="3d", facecolor="#000508")
    ax_mkt  = fig.add_axes([0.07, 0.04, 0.88, 0.25], facecolor="#000508")
    
    # ─── 3D SURFACE ───
    ax3d.plot_surface(
        X, Y, Z, cmap=cmap, edgecolor="none",
        alpha=0.82, rstride=2, cstride=2, antialiased=True,
    )
    
    # Floor contours
    try:
        ax3d.contour(X, Y, Z, levels=15, cmap=cmap, alpha=0.32,
                     offset=0, linewidths=0.5)
    except Exception:
        pass
    
    # Surface contours (mid-height)
    try:
        mid_z = float(np.median(Z))
        ax3d.contour(X, Y, Z, levels=10, colors="white", alpha=0.14,
                     linewidths=0.35, offset=mid_z)
    except Exception:
        pass
    
    # Dashed rings (VaR/CVaR boundaries)
    for radius, col, ls, al in [
        (bound * 0.38, "#ff1493", "--", 0.60),  # VaR boundary
        (bound * 0.65, "#8a2be2", ":",  0.40),  # CVaR boundary
    ]:
        theta = np.linspace(0, 2 * np.pi, 90)
        rx = radius * np.cos(theta)
        ry = radius * np.sin(theta)
        rz = compute_advanced_field(rx, ry, t, w, vol_prev)
        ax3d.plot(rx, ry, rz + 2.0, color=col, linestyle=ls,
                  linewidth=1.1, alpha=al, zorder=9)
    
    # ─── PARTICLE SCATTER ───
    trail_len = 35
    s0 = max(1, idx - trail_len)
    tx, ty = rets[s0:idx], rets[s0 - 1:idx - 1]
    if len(tx) > 0 and len(tx) == len(ty):
        tz = compute_advanced_field(tx, ty, t, w, vol_prev)
        sizes = 15 + 90 * np.abs(tx) / (np.max(np.abs(rets)) + 1e-12)
        colors = ["#00ff41" if r >= 0 else "#ff6600" for r in tx]
        alphas = np.linspace(0.25, 0.88, len(tx))
        for k in range(len(tx)):
            ax3d.scatter([tx[k]], [ty[k]], [tz[k] + 2.5],
                         c=[colors[k]], s=[sizes[k]], alpha=alphas[k],
                         edgecolors="white", linewidths=0.35, zorder=11)
    
    # Current return — large dot
    if 0 < idx < len(rets):
        cx, cy = rets[idx], rets[idx - 1]
        cz = compute_advanced_field(np.array([cx]), np.array([cy]), t, w, vol_prev)[0]
        cc = "#00ff41" if cx >= 0 else "#ff6600"
        ax3d.scatter([cx], [cy], [cz + 3.5], c=cc, s=250,
                     edgecolors="white", linewidths=1.5, zorder=16)
    
    # ─── CAMERA (FIXED) ───
    ax3d.view_init(elev=30, azim=-50)
    ax3d.set_zlim(0, 105)
    ax3d.set_xlim(-bound, bound)
    ax3d.set_ylim(-bound, bound)
    
    # Axis styling
    ax3d.set_xlabel("Φ(t) Return", color="#c0c0c0", fontsize=11, labelpad=7)
    ax3d.set_ylabel("Φ(t-1) Return", color="#c0c0c0", fontsize=11, labelpad=7)
    ax3d.set_zlabel("FIELD_DENSITY", color="#c0c0c0", fontsize=11, labelpad=5)
    ax3d.tick_params(colors="#909090", labelsize=7, pad=1)
    
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#1a1a20")
    
    ax3d.grid(color="#0f0f12", linewidth=0.3)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3d, shrink=0.45, aspect=22, pad=0.01,
                        location="right")
    cbar.set_label("PERIODICITY", color="white", fontsize=10, labelpad=6)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    
    # ─── TITLE + FORMULA ───
    date_str = dates[idx].strftime("%Y-%m-%d")
    regime_str = "CRISIS MODE" if regime_prob > 0.5 else "CALM"
    
    fig.text(0.50, 0.990,
             f"Advanced ζ-Field Quant Model  —  {sym}  |  {date_str}  |  ${prices[idx]:.2f}  |  {regime_str}",
             ha="center", va="top",
             color="white", fontsize=16, fontweight="bold",
             fontfamily="monospace",
             path_effects=[pe.withStroke(linewidth=3, foreground="black")])
    
    # Formula line 1
    fig.text(0.50, 0.965,
             "Ξₜ(x,y) = Norm₍₀,₁₀₀₎ [ V_Heston(κ,θ,ξ) + F_Regime(HMM) + C_Copula(ρ) + B_GARCH(α,β)",
             ha="center", va="top",
             color="#b8b8b8", fontsize=9.5,
             fontfamily="monospace",
             path_effects=[pe.withStroke(linewidth=1, foreground="black")])
    
    # Formula line 2
    fig.text(0.50, 0.948,
             "+ S_Lévy(α=1.7) + A_VarRiskPrem + H_HMM(Ω) + J_Merton(λ,μⱼ,σⱼ) + Q_VaR/CVaR + T_fBm(H=0.65) + Γ_Skew + Λ_Kurt ]",
             ha="center", va="top",
             color="#b8b8b8", fontsize=9.5,
             fontfamily="monospace",
             path_effects=[pe.withStroke(linewidth=1, foreground="black")])
    
    # Parameter display
    skew_val = (np.mean(w**3) / (vol**3 + 1e-12)) if len(w) >= 2 and vol > 0.01 else 0.0
    kurt_val = (np.mean(w**4) / (vol**4 + 1e-12)) if len(w) >= 2 and vol > 0.01 else 3.0
    
    param_str = (f"σ={vol:.2f}%  |  μ={mean_ret*100:.2f}%  |  "
                 f"Skew={skew_val:.2f}  |  Kurt={kurt_val:.1f}  |  Regime_P={regime_prob:.0%}")
    fig.text(0.50, 0.930,
             param_str,
             ha="center", va="top",
             color="#808080", fontsize=8,
             fontfamily="monospace")
    
    # ═══════════════════════════════════════
    # MARKET CHART (bottom)
    # ═══════════════════════════════════════
    ax_mkt.semilogy(dates[:idx + 1], bh[:idx + 1],
                    color="#888", linewidth=1.0, linestyle="--", alpha=0.85)
    ax_mkt.semilogy(dates[:idx + 1], eq_z[:idx + 1],
                    color="#00ddff", linewidth=1.8, alpha=0.92)
    ax_mkt.semilogy(dates[:idx + 1], eq_e[:idx + 1],
                    color="#c090ff", linewidth=1.6, alpha=0.92)
    
    # Future (faded)
    if idx < n - 1:
        ax_mkt.semilogy(dates[idx:], bh[idx:],
                        color="#888", linewidth=0.3, alpha=0.10)
    
    # Cursor
    ax_mkt.axvline(x=dates[idx], color="#ff3030", linewidth=1.6, alpha=0.75)
    
    # Drawdown shading
    peak_bh = np.maximum.accumulate(bh[:idx + 1])
    dd_bh = (peak_bh - bh[:idx + 1]) / peak_bh
    dd_mask = dd_bh > 0.10
    in_dd = False; dd_start = None
    for i in range(1, len(dd_mask)):
        if dd_mask[i] and not dd_mask[i - 1]:
            dd_start = dates[i]; in_dd = True
        if in_dd and (not dd_mask[i] or i == len(dd_mask) - 1):
            ax_mkt.axvspan(dd_start, dates[i], color="#cc2200", alpha=0.20)
            in_dd = False
    
    # Equity values
    bh_now,  z_now,  e_now = bh[idx], eq_z[idx], eq_e[idx]
    
    ax_mkt.text(0.01, 0.96,
                f"● BUY_&_HOLD  ${bh_now:.0f} ({(bh_now / 100 - 1)*100:+.0f}%)",
                transform=ax_mkt.transAxes, color="#888", fontsize=9,
                fontweight="bold", va="top")
    ax_mkt.text(0.01, 0.78,
                f"● ZETA_Advanced  ${z_now:.0f} ({(z_now / 100 - 1)*100:+.0f}%)",
                transform=ax_mkt.transAxes, color="#00ddff", fontsize=9,
                fontweight="bold", va="top")
    ax_mkt.text(0.01, 0.60,
                f"● ECLIPSE_Pro  ${e_now:.0f} ({(e_now / 100 - 1)*100:+.0f}%)",
                transform=ax_mkt.transAxes, color="#c090ff", fontsize=9,
                fontweight="bold", va="top")
    
    # Price labels
    for val, col in [(bh_now, "#888"), (z_now, "#00ddff"), (e_now, "#c090ff")]:
        ax_mkt.annotate(f"${val:.0f}", xy=(dates[idx], val),
                        xytext=(9, 0), textcoords="offset points",
                        color=col, fontsize=8, fontweight="bold", va="center")
    
    leader = "ZETA" if z_now >= e_now else "ECLIPSE"
    leader_val = max(z_now, e_now)
    ax_mkt.set_title(
        f"{date_str}  |  {leader} LEADS: ${leader_val:.0f}  |  Vol: {vol:.1f}%",
        color="#00ffcc", fontsize=11, fontweight="bold", pad=5)
    ax_mkt.set_ylabel("PORTFOLIO_VALUE", color="white", fontsize=10)
    ax_mkt.tick_params(colors="white", labelsize=8)
    ax_mkt.xaxis.set_major_formatter(mdates.DateFormatter("%Y.%m"))
    ax_mkt.grid(color="#14141a", linewidth=0.35)
    
    # Footer stats
    def _dd(eq):
        p = np.maximum.accumulate(eq[:idx + 1])
        return np.max((p - eq[:idx + 1]) / p) * 100
    
    def _sharpe(eq):
        if idx < 10:
            return 0.0
        rets_strat = np.diff(np.log(eq[:idx + 1]))
        return np.mean(rets_strat) / (np.std(rets_strat) + 1e-12) * np.sqrt(252)
    
    fig.text(0.50, 0.005,
             (f"DD: BH {_dd(bh):.1f}% | ZETA {_dd(eq_z):.1f}% | ECLIPSE {_dd(eq_e):.1f}%  •  "
              f"Sharpe: BH {_sharpe(bh):.2f} | ZETA {_sharpe(eq_z):.2f} | ECLIPSE {_sharpe(eq_e):.2f}"),
             ha="center", color="#505050", fontsize=7)
    
    # Render
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor="#000508",
                bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    from PIL import Image
    
    prices, dates, sym = fetch_prices()
    n = len(prices)
    rets = np.diff(np.log(prices)) * 100.0
    
    # Frame window
    start_idx = max(LOOKBACK, n - 130)
    frame_indices = np.linspace(start_idx, n - 1, N_FRAMES, dtype=int)
    
    # Grid
    ret_std = np.std(rets)
    bound = max(3.8 * ret_std, 5.0)
    lin = np.linspace(-bound, bound, N_GRID)
    X, Y = np.meshgrid(lin, lin)
    
    cmap = advanced_cmap()
    delay_ms = int(DURATION_S * 1000 / N_FRAMES)
    
    # Pre-compute vol history for GARCH
    vol_history = []
    for fi in range(N_FRAMES):
        idx = frame_indices[fi]
        w = rets[max(0, idx - LOOKBACK):idx]
        vol_history.append(np.std(w) if len(w) >= 2 else 1.0)
    
    # Pre-compute equity
    bh, eq_z, eq_e = precompute_equity(prices, rets, n)
    
    # Render frames
    frames = []
    print(f"\n[*] Rendering {N_FRAMES} frames (advanced 12-component model) …\n")
    
    for fi in range(N_FRAMES):
        if fi % 10 == 0:
            d = dates[frame_indices[fi]].strftime("%Y-%m-%d")
            print(f"    Frame {fi + 1:3d}/{N_FRAMES}  ({d})")
        
        buf = render_frame(fi, X, Y, lin, bound, prices, dates, rets, sym,
                           frame_indices, cmap, vol_history, bh, eq_z, eq_e)
        frames.append(Image.open(buf).convert("RGB"))
    
    # Save GIF
    print(f"\n[*] Saving advanced GIF ({len(frames)} frames) …")
    frames[0].save(
        SAVE_PATH, save_all=True, append_images=frames[1:],
        duration=delay_ms, loop=0,
    )
    sz = os.path.getsize(SAVE_PATH) / (1024 * 1024)
    print(f"[✓] {SAVE_PATH} ({sz:.1f} MB)\n")
    print("  ✓ 12-component advanced quant model")
    print("  ✓ Fixed camera axes")
    print("  ✓ Institutional-grade complexity\n")


if __name__ == "__main__":
    print("=" * 70)
    print("  Advanced ζ-Field — Institutional Quant Model")
    print("=" * 70)
    main()
