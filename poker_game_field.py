#!/usr/bin/env python3
"""
Poker Game Simulator with ζ-Field Decision Engine
===================================================
Simulates a full Texas Hold'em hand and uses the Information
Potential Field to analyze each decision point in real-time.

Outputs a premium animated GIF showing:
  TOP:     3D strategy field surface (equity × pot_odds → action density)
  MIDDLE:  Card table — hero hand, community cards, pot, opponent range
  BOTTOM:  Decision timeline + EV analysis

The field morphs at every street as new information arrives.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from PIL import Image
import io
import os
import warnings

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════
N_GRID     = 90
DPI        = 105
FIG_W      = 13
FIG_H      = 16
SAVE_PATH  = "poker_game_analysis.gif"

# ═══════════════════════════════════════════
# CARD ENGINE
# ═══════════════════════════════════════════
RANKS = "23456789TJQKA"
SUITS = "♠♥♦♣"
SUIT_COLORS = {"♠": "#e0e0e0", "♣": "#90ee90", "♦": "#6ec6ff", "♥": "#ff6666"}

def card_str(c):
    return RANKS[c % 13] + SUITS[c // 13]

def card_rank(c):
    return c % 13

def card_suit(c):
    return c // 13

def hand_rank_simple(hole, board):
    """Simplified hand ranking (0-8) for demo. Uses pair/trip/straight heuristics."""
    all_cards = list(hole) + list(board)
    ranks = sorted([card_rank(c) for c in all_cards], reverse=True)
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    
    counts = sorted(rank_counts.values(), reverse=True)
    
    # Check flush
    suits = [card_suit(c) for c in all_cards]
    suit_counts = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    is_flush = max(suit_counts.values()) >= 5 if suit_counts else False
    
    # Check straight
    unique_ranks = sorted(set(ranks))
    is_straight = False
    for i in range(len(unique_ranks) - 4):
        if unique_ranks[i + 4] - unique_ranks[i] == 4:
            is_straight = True
    # Wheel (A-2-3-4-5)
    if set([12, 0, 1, 2, 3]).issubset(set(ranks)):
        is_straight = True
    
    if is_straight and is_flush:
        return 8  # Straight flush
    if counts[0] == 4:
        return 7  # Quads
    if counts[0] == 3 and len(counts) > 1 and counts[1] >= 2:
        return 6  # Full house
    if is_flush:
        return 5  # Flush
    if is_straight:
        return 4  # Straight
    if counts[0] == 3:
        return 3  # Trips
    if counts[0] == 2 and len(counts) > 1 and counts[1] == 2:
        return 2  # Two pair
    if counts[0] == 2:
        return 1  # One pair
    return 0  # High card

HAND_NAMES = [
    "High Card", "One Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"
]

def monte_carlo_equity(hole, board, n_sims=2000):
    """Estimate equity vs a random opponent hand via Monte Carlo."""
    deck = [c for c in range(52) if c not in hole and c not in board]
    wins = ties = 0
    
    for _ in range(n_sims):
        d = deck.copy()
        np.random.shuffle(d)
        
        # Deal opponent
        opp = [d[0], d[1]]
        idx = 2
        
        # Complete board
        remaining = 5 - len(board)
        run_board = list(board) + list(d[idx:idx + remaining])
        
        hero_rank = hand_rank_simple(hole, run_board)
        opp_rank = hand_rank_simple(opp, run_board)
        
        if hero_rank > opp_rank:
            wins += 1
        elif hero_rank == opp_rank:
            ties += 1
    
    return (wins + ties * 0.5) / n_sims * 100.0


# ═══════════════════════════════════════════
# FIELD COMPUTATION
# ═══════════════════════════════════════════
def poker_field(E, P, equity, pot_odds, street, aggr, hand_strength):
    """
    Information Potential Field Ψ(e, p).
    
    Components:
      V_equity:    Equity-based value surface
      B_bluff:     Bluff frequency (low equity, polarized)
      R_raise:     Raise incentive (high equity peak)
      C_call:      Calling range (medium equity, good pot odds)
      F_fold:      Fold pressure (low equity, bad odds)
      S_street:    Street-dependent aggression scaling
      J_jump:      Discontinuity at key equity thresholds
      Γ_position:  Position-aware asymmetry
    """
    e = E / 100.0
    p = P / 100.0
    
    eq = equity / 100.0
    po = pot_odds / 100.0
    
    # V: Core value surface (Gaussian peak at current equity)
    V = 65 * np.exp(-3.0 * (e - eq)**2 - 1.5 * (p - po)**2)
    
    # R: Raise incentive (strong hands)
    R = 50 * e**2.5 * np.exp(-2.0 * (p - 0.4)**2) * (1 + 0.3 * street)
    
    # C: Call surface (medium hands, good pot odds)
    C = 35 * np.exp(-5.0 * (e - 0.45)**2) * (1 - p) * np.exp(-1.0 * (p - po)**2)
    
    # F: Fold pressure (low equity, bad pot odds)
    F = -20 * (1 - e)**2 * p * np.exp(-0.5 * e**2)
    
    # B: Bluff frequency (low equity but polarized betting)
    B = 18 * (1 - e)**2.5 * aggr * np.exp(-3.0 * (p - 0.6)**2)
    
    # S: Street scaling (more decisive on later streets)
    S = 8 * street * e * (1 - e) * np.sin(np.pi * p)
    
    # J: Jump at key thresholds (pot-odds break-even)
    breakeven = po
    J = 15 * np.tanh(8.0 * (e - breakeven)) * np.exp(-2.0 * (p - po)**2)
    
    # Γ: Hand strength asymmetry
    G = 10 * hand_strength * e * np.exp(-0.5 * (p - 0.5)**2)
    
    field = V + R + C + F + B + S + J + G
    
    # Normalize to [0, 100]
    fmin, fmax = field.min(), field.max()
    if fmax - fmin < 1e-8:
        return np.full_like(field, 50.0)
    return 100.0 * (field - fmin) / (fmax - fmin)


def get_recommendation(equity, pot_odds, street, hand_str):
    """Get action recommendation + EV estimate."""
    eq = equity / 100.0
    po = pot_odds / 100.0
    
    # Expected value of calling
    ev_call = eq * (1 + 1/max(po, 0.01)) - 1
    
    # Threshold logic combining equity, pot odds, street, hand strength
    if eq > 0.65 + 0.05 * street:
        action = "RAISE"
        confidence = min(99, int(60 + 40 * eq))
        ev = ev_call * 1.5
    elif eq > po - 0.05 and eq > 0.30:
        action = "CALL"
        confidence = min(95, int(40 + 50 * eq))
        ev = ev_call
    elif eq < 0.25 and np.random.random() < 0.15 * (1 + street * 0.1):
        action = "BLUFF RAISE"
        confidence = int(30 + 20 * np.random.random())
        ev = -0.3 + 0.5 * (1 - eq)
    else:
        action = "FOLD"
        confidence = min(95, int(50 + 40 * (1 - eq)))
        ev = 0.0
    
    return action, confidence, ev


# ═══════════════════════════════════════════
# COLORMAP
# ═══════════════════════════════════════════
def field_cmap():
    stops = [
        (0.00, "#04041a"), (0.12, "#0a1268"), (0.25, "#1650a0"),
        (0.38, "#2090c0"), (0.50, "#30c0c8"), (0.62, "#60dca0"),
        (0.75, "#a0f070"), (0.87, "#e8d020"), (1.00, "#c01010"),
    ]
    return LinearSegmentedColormap.from_list("pf", stops)


# ═══════════════════════════════════════════
# EXAMPLE HAND — a realistic scenario
# ═══════════════════════════════════════════
def create_example_hand():
    """
    Simulated hand: Hero has A♠K♠ in a classic spot.
    We play through preflop → flop → turn → river.
    """
    # Card encoding: rank + suit*13
    # Suits: 0=♠, 1=♥, 2=♦, 3=♣
    # Ranks: 0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A
    
    A_s = 12 + 0*13   # A♠
    K_s = 11 + 0*13   # K♠
    
    hero_hand = [A_s, K_s]
    
    # Board cards (revealed progressively)
    board_flop  = [8 + 0*13,  6 + 1*13,  2 + 2*13]   # T♠ 8♥ 4♦
    board_turn  = [10 + 0*13]                            # Q♠ (flush draw!)
    board_river = [0 + 3*13]                             # 2♣
    
    # Opponent: K♥ J♦ (eventually loses)
    opp_hand = [11 + 1*13, 9 + 2*13]
    
    # Streets with game state
    streets = [
        {
            "name": "PRE-FLOP",
            "board": [],
            "pot": 15,        # blinds
            "bet_to_call": 10,
            "opp_action": "RAISES to $10",
            "stack_hero": 490,
            "stack_opp": 490,
            "n_frames": 8,
        },
        {
            "name": "FLOP",
            "board": board_flop,
            "pot": 35,
            "bet_to_call": 20,
            "opp_action": "BETS $20 (57% pot)",
            "stack_hero": 480,
            "stack_opp": 470,
            "n_frames": 12,
        },
        {
            "name": "TURN",
            "board": board_flop + board_turn,
            "pot": 75,
            "bet_to_call": 45,
            "opp_action": "BETS $45 (60% pot)",
            "stack_hero": 460,
            "stack_opp": 450,
            "n_frames": 12,
        },
        {
            "name": "RIVER",
            "board": board_flop + board_turn + board_river,
            "pot": 165,
            "bet_to_call": 80,
            "opp_action": "BETS $80 (48% pot)",
            "stack_hero": 415,
            "stack_opp": 405,
            "n_frames": 12,
        },
    ]
    
    return hero_hand, opp_hand, streets


# ═══════════════════════════════════════════
# DRAW CARD (on the matplotlib canvas)
# ═══════════════════════════════════════════
def draw_card(ax, x, y, card_idx, width=0.08, height=0.11, facedown=False):
    """Draw a playing card on the axes."""
    if facedown:
        rect = mpatches.FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.008",
            facecolor="#1a3060", edgecolor="#556688",
            linewidth=1.5, transform=ax.transAxes
        )
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, "?",
                ha="center", va="center", fontsize=18,
                color="#8090a0", fontweight="bold",
                transform=ax.transAxes)
        return
    
    rank = RANKS[card_idx % 13]
    suit = SUITS[card_idx // 13]
    suit_color = SUIT_COLORS[suit]
    
    bg = "#f8f8f0" if suit in ("♦", "♥") else "#f0f0f8"
    edge = "#cc3333" if suit in ("♦", "♥") else "#333333"
    
    rect = mpatches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.008",
        facecolor=bg, edgecolor=edge,
        linewidth=1.8, transform=ax.transAxes
    )
    ax.add_patch(rect)
    
    text_color = "#c02020" if suit in ("♦", "♥") else "#202020"
    
    ax.text(x + width/2, y + height * 0.62, rank,
            ha="center", va="center", fontsize=20,
            color=text_color, fontweight="bold",
            transform=ax.transAxes)
    ax.text(x + width/2, y + height * 0.28, suit,
            ha="center", va="center", fontsize=16,
            color=text_color, transform=ax.transAxes)


# ═══════════════════════════════════════════
# RENDER FRAME
# ═══════════════════════════════════════════
def render_frame(frame_data, X, Y, cmap):
    """
    Render one frame of the poker game analysis.
    
    Layout:
      [0.38 - 0.98]  3D field surface
      [0.20 - 0.38]  Card table + game info
      [0.02 - 0.20]  Decision timeline / EV chart
    """
    street     = frame_data["street"]
    hero_hand  = frame_data["hero_hand"]
    board      = frame_data["board"]
    pot        = frame_data["pot"]
    bet_to_call= frame_data["bet_to_call"]
    equity     = frame_data["equity"]
    pot_odds   = frame_data["pot_odds"]
    action     = frame_data["action"]
    confidence = frame_data["confidence"]
    ev         = frame_data["ev"]
    hand_name  = frame_data["hand_name"]
    hand_str   = frame_data["hand_str"]
    street_idx = frame_data["street_idx"]
    opp_action = frame_data["opp_action"]
    phase      = frame_data["phase"]
    timeline   = frame_data["timeline"]
    stack_hero = frame_data["stack_hero"]
    stack_opp  = frame_data["stack_opp"]
    opp_hand   = frame_data["opp_hand"]
    is_showdown= frame_data.get("is_showdown", False)
    
    # Aggression estimate
    aggr = bet_to_call / max(pot, 1)
    
    # Compute field
    Z = poker_field(X, Y, equity, pot_odds, street_idx, aggr, hand_str / 8.0)
    
    # ── FIGURE ──
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#000810")
    
    ax3d   = fig.add_axes([0.02, 0.40, 0.96, 0.56], projection="3d", facecolor="#000810")
    ax_tbl = fig.add_axes([0.02, 0.18, 0.96, 0.22], facecolor="#000810")
    ax_ev  = fig.add_axes([0.08, 0.03, 0.84, 0.13], facecolor="#000810")
    
    # ════════════════════════════════════
    # 3D FIELD SURFACE
    # ════════════════════════════════════
    ax3d.plot_surface(X, Y, Z, cmap=cmap, edgecolor="none",
                      alpha=0.82, rstride=2, cstride=2, antialiased=True)
    
    # Floor contours
    try:
        ax3d.contour(X, Y, Z, levels=10, cmap=cmap, alpha=0.22,
                     offset=0, linewidths=0.4)
    except:
        pass
    
    # Decision point marker
    z_at_point = poker_field(
        np.array([[equity]]), np.array([[pot_odds]]),
        equity, pot_odds, street_idx, aggr, hand_str / 8.0
    )[0, 0]
    
    marker_color = {"RAISE": "#ff2020", "CALL": "#00ff60",
                    "FOLD": "#4080ff", "BLUFF RAISE": "#ff8000",
                    "CHECK": "#00ff60"}.get(action, "#ffffff")
    
    ax3d.scatter([equity], [pot_odds], [z_at_point + 5],
                 c=marker_color, s=350, edgecolors="white",
                 linewidths=2.5, zorder=20, alpha=0.95)
    
    # Break-even line
    be_line = np.linspace(0, 100, 50)
    be_z = poker_field(
        be_line.reshape(1, -1) * np.ones((1, 50)),
        be_line.reshape(1, -1) * np.ones((1, 50)),
        equity, pot_odds, street_idx, aggr, hand_str / 8.0
    )
    ax3d.plot(be_line, be_line, be_z[0] + 2,
              color="#ff00ff", linewidth=1.5, alpha=0.5,
              linestyle="--", label="Break-even")
    
    # Camera & axes
    ax3d.view_init(elev=28, azim=-55)
    ax3d.set_xlim(0, 100)
    ax3d.set_ylim(0, 100)
    ax3d.set_zlim(0, 105)
    
    ax3d.set_xlabel("HAND EQUITY (%)", color="#909090", fontsize=10, labelpad=6)
    ax3d.set_ylabel("POT ODDS (%)", color="#909090", fontsize=10, labelpad=6)
    ax3d.set_zlabel("Ψ DENSITY", color="#909090", fontsize=10, labelpad=4)
    ax3d.tick_params(colors="#606060", labelsize=7, pad=1)
    
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#181820")
    ax3d.grid(color="#101018", linewidth=0.3)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3d, shrink=0.40, aspect=20, pad=0.01)
    cbar.set_label("ACTION WEIGHT", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    
    # ── Title ──
    fig.text(0.50, 0.975,
             f"ζ-FIELD POKER ENGINE  —  {street}",
             ha="center", va="top", color="white", fontsize=17,
             fontweight="bold", fontfamily="monospace",
             path_effects=[pe.withStroke(linewidth=3, foreground="black")])
    
    fig.text(0.50, 0.955,
             f"Ψ(e,p) = V_Equity + R_Raise + C_Call + F_Fold + B_Bluff + S_Street + J_Jump + Γ_HandStr",
             ha="center", va="top", color="#707070", fontsize=8.5,
             fontfamily="monospace")
    
    # ════════════════════════════════════
    # CARD TABLE (middle section)
    # ════════════════════════════════════
    ax_tbl.set_xlim(0, 1)
    ax_tbl.set_ylim(0, 1)
    ax_tbl.axis("off")
    
    # Table felt background
    felt = mpatches.FancyBboxPatch(
        (0.02, 0.05), 0.96, 0.90,
        boxstyle="round,pad=0.02",
        facecolor="#0a3020", edgecolor="#1a5038",
        linewidth=2, transform=ax_tbl.transAxes
    )
    ax_tbl.add_patch(felt)
    
    # Hero hand label
    ax_tbl.text(0.12, 0.85, "HERO", ha="center", va="center",
                color="#00ff80", fontsize=12, fontweight="bold",
                fontfamily="monospace", transform=ax_tbl.transAxes)
    
    # Draw hero cards
    draw_card(ax_tbl, 0.05, 0.40, hero_hand[0])
    draw_card(ax_tbl, 0.15, 0.40, hero_hand[1])
    
    # Board cards
    ax_tbl.text(0.50, 0.85, "COMMUNITY", ha="center", va="center",
                color="#d0d080", fontsize=11, fontweight="bold",
                fontfamily="monospace", transform=ax_tbl.transAxes)
    
    board_x_start = 0.30
    for i in range(5):
        x = board_x_start + i * 0.09
        if i < len(board):
            draw_card(ax_tbl, x, 0.40, board[i])
        else:
            draw_card(ax_tbl, x, 0.40, 0, facedown=True)
    
    # Opponent
    ax_tbl.text(0.88, 0.85, "VILLAIN", ha="center", va="center",
                color="#ff6060", fontsize=12, fontweight="bold",
                fontfamily="monospace", transform=ax_tbl.transAxes)
    
    if is_showdown:
        draw_card(ax_tbl, 0.81, 0.40, opp_hand[0])
        draw_card(ax_tbl, 0.91, 0.40, opp_hand[1])
    else:
        draw_card(ax_tbl, 0.81, 0.40, 0, facedown=True)
        draw_card(ax_tbl, 0.91, 0.40, 0, facedown=True)
    
    # Pot + stacks
    ax_tbl.text(0.50, 0.28, f"POT: ${pot}",
                ha="center", va="center", color="#ffd700",
                fontsize=15, fontweight="bold", fontfamily="monospace",
                transform=ax_tbl.transAxes,
                path_effects=[pe.withStroke(linewidth=2, foreground="#3a2800")])
    
    ax_tbl.text(0.12, 0.18, f"Stack: ${stack_hero}",
                ha="center", va="center", color="#80c0a0",
                fontsize=9, fontfamily="monospace",
                transform=ax_tbl.transAxes)
    ax_tbl.text(0.88, 0.18, f"Stack: ${stack_opp}",
                ha="center", va="center", color="#c08080",
                fontsize=9, fontfamily="monospace",
                transform=ax_tbl.transAxes)
    
    # Opponent action
    ax_tbl.text(0.50, 0.12, f"Villain {opp_action}",
                ha="center", va="center", color="#ff9060",
                fontsize=10, fontfamily="monospace",
                transform=ax_tbl.transAxes)
    
    # ── RECOMMENDATION PANEL ──
    action_bg_color = {"RAISE": "#400808", "CALL": "#083808",
                       "FOLD": "#080838", "BLUFF RAISE": "#382008",
                       "CHECK": "#083808"}.get(action, "#181818")
    
    rec_bg = mpatches.FancyBboxPatch(
        (0.28, 0.58), 0.44, 0.38,
        boxstyle="round,pad=0.01",
        facecolor=action_bg_color, edgecolor=marker_color,
        linewidth=2.5, alpha=0.85, transform=ax_tbl.transAxes
    )
    ax_tbl.add_patch(rec_bg)
    
    ax_tbl.text(0.50, 0.88,
                f"▶ {action}",
                ha="center", va="center", color=marker_color,
                fontsize=18, fontweight="bold", fontfamily="monospace",
                transform=ax_tbl.transAxes,
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])
    
    ax_tbl.text(0.50, 0.72,
                f"Equity: {equity:.1f}%  |  Pot Odds: {pot_odds:.1f}%  |  EV: {ev:+.2f}x",
                ha="center", va="center", color="#c0c0c0",
                fontsize=9, fontfamily="monospace",
                transform=ax_tbl.transAxes)
    
    ax_tbl.text(0.50, 0.62,
                f"Hand: {hand_name}  |  Confidence: {confidence}%",
                ha="center", va="center", color="#909090",
                fontsize=8.5, fontfamily="monospace",
                transform=ax_tbl.transAxes)
    
    # ════════════════════════════════════
    # DECISION TIMELINE (bottom)
    # ════════════════════════════════════
    street_labels = ["PRE", "FLOP", "TURN", "RIVER"]
    street_colors_map = {"RAISE": "#ff2020", "CALL": "#00ff60",
                         "FOLD": "#4080ff", "BLUFF RAISE": "#ff8000",
                         "CHECK": "#00ff60"}
    
    ax_ev.set_xlim(-0.5, 3.5)
    ax_ev.set_ylim(-0.5, 1.5)
    ax_ev.set_xticks(range(4))
    ax_ev.set_xticklabels(street_labels, color="#c0c0c0", fontsize=10,
                          fontfamily="monospace")
    ax_ev.set_yticks([])
    ax_ev.spines["top"].set_visible(False)
    ax_ev.spines["right"].set_visible(False)
    ax_ev.spines["left"].set_visible(False)
    ax_ev.spines["bottom"].set_color("#303040")
    ax_ev.tick_params(colors="#808080")
    
    ax_ev.set_title("DECISION TIMELINE  |  ζ-Field Analysis",
                    color="#00ccaa", fontsize=11, fontweight="bold",
                    fontfamily="monospace", pad=6)
    
    for i, entry in enumerate(timeline):
        col = street_colors_map.get(entry["action"], "#808080")
        
        # Circle marker
        ax_ev.scatter([i], [0.8], s=500, c=col, edgecolors="white",
                      linewidths=2, zorder=10, alpha=0.9)
        
        # Action text inside circle
        short = entry["action"][:1]
        ax_ev.text(i, 0.8, short, ha="center", va="center",
                   color="white", fontsize=14, fontweight="bold",
                   fontfamily="monospace", zorder=11)
        
        # Equity below
        ax_ev.text(i, 0.25, f"{entry['equity']:.0f}%",
                   ha="center", va="center", color="#a0a0a0",
                   fontsize=10, fontfamily="monospace")
        
        # EV below that
        ev_col = "#00ff60" if entry["ev"] >= 0 else "#ff4040"
        ax_ev.text(i, -0.15, f"EV: {entry['ev']:+.2f}x",
                   ha="center", va="center", color=ev_col,
                   fontsize=8.5, fontfamily="monospace")
        
        # Connection line
        if i > 0:
            prev_col = street_colors_map.get(timeline[i-1]["action"], "#808080")
            ax_ev.plot([i-1, i], [0.8, 0.8], color="#404050",
                       linewidth=2, zorder=5)
    
    # Highlight current street
    if street_idx < len(timeline):
        ax_ev.scatter([street_idx], [0.8], s=700, facecolors="none",
                      edgecolors="#ffffff", linewidths=3, zorder=12)
    
    # ── Render ──
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor="#000810",
                bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    print("\n" + "=" * 70)
    print("  ζ-Field Poker Game Engine — Live Hand Analysis")
    print("=" * 70 + "\n")
    
    hero_hand, opp_hand, streets = create_example_hand()
    
    print(f"  HERO:    {card_str(hero_hand[0])} {card_str(hero_hand[1])}")
    print(f"  VILLAIN: {card_str(opp_hand[0])} {card_str(opp_hand[1])} (hidden)\n")
    
    # Grid
    lin = np.linspace(0, 100, N_GRID)
    X, Y = np.meshgrid(lin, lin)
    cmap = field_cmap()
    
    frames = []
    timeline = []
    
    np.random.seed(42)
    
    for si, st in enumerate(streets):
        # Compute equity
        equity = monte_carlo_equity(hero_hand, st["board"])
        pot_odds = st["bet_to_call"] / (st["pot"] + st["bet_to_call"]) * 100 if st["bet_to_call"] > 0 else 0
        hand_str = hand_rank_simple(hero_hand, st["board"]) if st["board"] else 0
        hand_name = HAND_NAMES[hand_str] if st["board"] else "Unpaired AK"
        
        action, confidence, ev = get_recommendation(equity, pot_odds, si, hand_str)
        
        print(f"  [{st['name']:10s}] Equity: {equity:.1f}%  Pot Odds: {pot_odds:.1f}%  "
              f"→ {action} ({confidence}% conf)  EV: {ev:+.2f}x")
        if st["board"]:
            board_str = " ".join(card_str(c) for c in st["board"])
            print(f"             Board: {board_str}  |  Hand: {hand_name}")
        
        timeline.append({
            "action": action,
            "equity": equity,
            "ev": ev,
        })
        
        # Is this the last street? → showdown on final frames
        is_last = (si == len(streets) - 1)
        
        for fi in range(st["n_frames"]):
            # Animate field - slight evolution within the street
            phase = fi / st["n_frames"]
            
            # Determine if showdown (last 3 frames of river)
            is_showdown = is_last and fi >= st["n_frames"] - 3
            
            frame_data = {
                "street": st["name"],
                "hero_hand": hero_hand,
                "opp_hand": opp_hand,
                "board": st["board"],
                "pot": st["pot"],
                "bet_to_call": st["bet_to_call"],
                "equity": equity + 1.5 * np.sin(phase * np.pi),  # slight animation
                "pot_odds": pot_odds,
                "action": action if not is_showdown else "SHOWDOWN",
                "confidence": confidence,
                "ev": ev,
                "hand_name": hand_name if not is_showdown else f"{hand_name} — HERO WINS!" if equity > 50 else f"{hand_name}",
                "hand_str": hand_str,
                "street_idx": si,
                "opp_action": st["opp_action"],
                "phase": phase,
                "timeline": timeline[:si + 1],
                "stack_hero": st["stack_hero"],
                "stack_opp": st["stack_opp"],
                "is_showdown": is_showdown,
            }
            
            buf = render_frame(frame_data, X, Y, cmap)
            frames.append(Image.open(buf).convert("RGB"))
        
        print()
    
    # Save GIF
    total_frames = len(frames)
    print(f"[*] Saving {SAVE_PATH} ({total_frames} frames) …")
    
    # Variable frame duration: hold longer on street transitions
    durations = []
    frame_idx = 0
    for st in streets:
        for fi in range(st["n_frames"]):
            if fi == 0:
                durations.append(800)  # 800ms pause at street start
            elif fi == st["n_frames"] - 1:
                durations.append(600)
            else:
                durations.append(200)
            frame_idx += 1
    
    frames[0].save(
        SAVE_PATH, save_all=True, append_images=frames[1:],
        duration=durations, loop=0,
    )
    
    size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)
    print(f"[✓] {SAVE_PATH} ({size_mb:.1f} MB)\n")
    print("  ✓ Live poker game analysis with ζ-field engine")
    print("  ✓ Real-time strategy field evolving per street")
    print("  ✓ Decision timeline with EV tracking")
    print("  ✓ Card display + opponent action overlay")
    print()
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
