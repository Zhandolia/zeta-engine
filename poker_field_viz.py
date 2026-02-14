#!/usr/bin/env python3
"""
Poker Information Field — 3D Visualization GIF
===============================================
Demonstrates the poker strategy field concept using a hand evolution.

Shows how the optimal action probability surface changes as:
  • Hand equity evolves (cards are revealed)
  • Pot odds change (betting progresses)
  • Opponent range narrows

X-axis: Hand Equity (0-100%)
Y-axis: Pot Odds / Implied Odds (0-100%)
Z-axis: Action Probability Density (Fold=blue, Call=green, Raise=red peaks)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from PIL import Image
import io


# Config
N_GRID = 100
N_FRAMES = 60
DPI = 100
SAVE_PATH = "poker_field.gif"


def poker_action_field(equity, pot_odds, street, aggression):
    """
    Compute poker action probability field.
    
    Returns fold/call/raise probabilities based on:
      - equity: hand strength (0-100%)
      - pot_odds: pot price (0-100%)
      - street: 0=preflop, 1=flop, 2=turn, 3=river
      - aggression: opponent aggression level
    """
    e = equity / 100.0
    p = pot_odds / 100.0
    
    # GTO baseline: value-based strategy
    # High equity → Raise
    # Medium equity → Call if pot odds favorable
    # Low equity → Fold (but some bluffs)
    
    # Raise surface (peaks at high equity)
    raise_surface = 80 * e**2 * np.exp(-0.5 * (p - 0.5)**2)
    
    # Call surface (medium equity, good pot odds)
    call_surface = 60 * (e * (1 - e)) * (1 - p) * np.exp(-0.3 * (e - 0.5)**2)
    
    # Fold surface (low equity, bad pot odds)
    fold_surface = 40 * (1 - e) * p * np.exp(-0.2 * e**2)
    
    # Bluff component (low equity but aggressive)
    bluff = 25 * (1 - e)**2 * aggression * np.exp(-1.5 * (p - 0.7)**2)
    
    # Street adjustment (more aggressive later streets)
    street_factor = 1.0 + 0.3 * street
    
    # Combined field
    field = (raise_surface + bluff * street_factor + 
             call_surface * 0.8 + fold_surface * 0.5)
    
    return field


def render_frame(frame_idx, X, Y, lin):
    """Render a single frame showing field evolution during a hand."""
    
    # Simulate hand progression
    # Preflop → Flop → Turn → River
    progress = frame_idx / N_FRAMES
    
    #  hand equity increases as we see community cards
    if progress < 0.25:
        # Preflop: uncertain
        street = 0
        avg_equity = 35 + 30 * progress / 0.25
        equity_volatility = 25
    elif progress < 0.50:
        # Flop
        street = 1
        avg_equity = 65
        equity_volatility = 20
    elif progress < 0.75:
        # Turn
        street = 2
        avg_equity = 70
        equity_volatility = 15
    else:
        # River
        street = 3
        avg_equity = 75
        equity_volatility = 10
    
    # Pot odds increase as more betting occurs
    avg_pot_odds = min(70, 20 + progress * 60)
    
    # Opponent aggression (dynamic)
    aggression = 0.3 + 0.4 * np.sin(progress * 2 * np.pi)
    
    # Equity distribution (gaussian around current hand strength)
    equity_dist = avg_equity + equity_volatility * np.tanh((X - 50) / 30)
    pot_odds_dist = avg_pot_odds + 20 * np.tanh((Y - 50) / 30)
    
    # Compute field
    Z = poker_action_field(equity_dist, pot_odds_dist, street, aggression)
    
    # Create figure
    fig = plt.figure(figsize=(14, 10), facecolor='#000810')
    ax = fig.add_subplot(111, projection='3d', facecolor='#000810')
    
    # Colormap (cold=fold, warm=raise)
    colors = [
        (0.00, "#05051a"), (0.20, "#1040a0"), (0.40, "#20a0c0"),
        (0.55, "#40d080"), (0.70, "#d0d040"), (0.85, "#f08030"),
        (1.00, "#c00808"),
    ]
    cmap = LinearSegmentedColormap.from_list("poker", colors)
    
    # Surface
    ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none',
                    alpha=0.85, rstride=2, cstride=2, antialiased=True)
    
    # Floor contours
    try:
        ax.contour(X, Y, Z, levels=12, cmap=cmap, alpha=0.25,
                   offset=0, linewidths=0.4)
    except:
        pass
    
    # Current decision point
    current_equity = avg_equity
    current_pot_odds = avg_pot_odds
    current_z = poker_action_field(
        np.array([current_equity]), 
        np.array([current_pot_odds]),
        street, aggression
    )[0]
    
    ax.scatter([current_equity], [current_pot_odds], [current_z + 5],
               c='#00ff00', s=300, edgecolors='white', linewidths=2.5,
               marker='o', zorder=20, alpha=0.95)
    
    # Camera
    ax.view_init(elev=28, azim=-60)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)
    
    # Axes
    ax.set_xlabel("HAND EQUITY (%)", color="#b0b0b0", fontsize=11, labelpad=8)
    ax.set_ylabel("POT ODDS (%)", color="#b0b0b0", fontsize=11, labelpad=8)
    ax.set_zlabel("ACTION DENSITY", color="#b0b0b0", fontsize=11, labelpad=6)
    ax.tick_params(colors="#707070", labelsize=8)
    
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#202030")
    
    ax.grid(color="#15151c", linewidth=0.3)
    
    # Title
    street_names = ["PRE-FLOP", "FLOP", "TURN", "RIVER"]
    title = f"Poker Information Field  —  {street_names[street]}"
    
    fig.text(0.50, 0.96, title,
             ha="center", color="white", fontsize=16, fontweight="bold",
             fontfamily="monospace",
             path_effects=[pe.withStroke(linewidth=3, foreground="black")])
    
    # Info display
    action_desc = "RAISE" if current_z > 60 else ("CALL" if current_z > 30 else "FOLD")
    action_color = "#ff4040" if action_desc == "RAISE" else ("#40ff40" if action_desc == "CALL" else "#4040ff")
    
    info_text = (f"Equity: {current_equity:.0f}%  |  Pot Odds: {current_pot_odds:.0f}%  |  "
                 f"Aggression: {aggression:.2f}  |  Recommended: {action_desc}")
    
    fig.text(0.50, 0.92, info_text,
             ha="center", color="#a0a0a0", fontsize=10,
             fontfamily="monospace")
    
    # Formula
    formula = "Ψ(e,p) = V_GTO(e,p) + B_Bluff(e,p,α) + R_Regime(street) × [1 + ξ_Exploit(opp)]"
    fig.text(0.50, 0.02, formula,
             ha="center", color="#606060", fontsize=8,
             fontfamily="monospace", style="italic")
    
    # Render
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, facecolor='#000810',
                bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    buf.seek(0)
    
    return buf


def main():
    print("\n" + "=" * 70)
    print("  Poker Information Field — GIF Generator")
    print("=" * 70 + "\n")
    
    # Grid
    lin = np.linspace(0, 100, N_GRID)
    X, Y = np.meshgrid(lin, lin)
    
    # Render frames
    frames = []
    print(f"[*] Rendering {N_FRAMES} frames …\n")
    
    for i in range(N_FRAMES):
        if i % 10 == 0:
            print(f"    Frame {i+1:3d}/{N_FRAMES}")
        
        buf = render_frame(i, X, Y, lin)
        frames.append(Image.open(buf).convert("RGB"))
    
    # Save GIF
    print(f"\n[*] Saving {SAVE_PATH} …")
    delay_ms = 150  # ~6.7 fps
    
    frames[0].save(
        SAVE_PATH, save_all=True, append_images=frames[1:],
        duration=delay_ms, loop=0
    )
    
    import os
    size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)
    
    print(f"[✓] {SAVE_PATH} ({size_mb:.1f} MB)")
    print(f"\n  ✓ Poker strategy field visualization ready!")
    print(f"  ✓ Shows hand evolution: Preflop → Flop → Turn → River")
    print(f"  ✓ Action recommendation updates in real-time\n")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
