# ζ-Field Poker AI Engine

> 8-component game-theoretic field mapping equity × pot-odds space. Monte Carlo simulation feeds a continuous decision surface — play against it live.

![ζ-Field Poker Engine](portfolio/assets/poker_game_analysis.gif)

## 🎯 Live Demo

Open `portfolio/index.html` in your browser to:
- **Play Texas Hold'em** against the AI with real-time field analysis
- **Explore the Model** — formula breakdown, architecture pipeline, game theory foundations
- **Watch the heatmap** shift as each card is dealt and every bet is placed

## 🧮 The Model

The engine evaluates every decision through an 8-component continuous field:

```
Ψ(e, p) = V_equity + R_raise + C_call + F_fold + B_bluff + S_street + J_jump + Γ_strength
```

| Component | Force | Description |
|-----------|-------|-------------|
| **V** | Equity Peak | Gaussian centered on hero's equity position |
| **R** | Raise Pressure | Aggression surface with street multiplier |
| **C** | Call Defense | Defensive ridge at ~45% equity |
| **F** | Fold Gravity | Negative basin pulling weak hands to fold |
| **B** | Bluff Surface | Deception peaks for profitable bluff spots |
| **J** | Jump Threshold | Sharp sigmoid modeling decision boundaries |

## 🏗 Architecture

```
Monte Carlo (2,400 rollouts) → ζ-Field Eval (8 components) → Decision (RAISE/CALL/FOLD)
```

## 📁 Project Structure

```
broki/
├── portfolio/              # Web application
│   ├── index.html          # Main page — hero, model deep-dive, poker game, about
│   ├── style.css           # Full dark-theme styling with glassmorphism
│   ├── main.js             # Hero canvas animation, scroll effects, nav
│   ├── poker/
│   │   ├── engine.js       # Card evaluation, Monte Carlo simulation, hand ranking
│   │   ├── field.js        # ζ-field computation, commentary, heatmap rendering
│   │   ├── game.js         # Game state machine, AI opponent logic
│   │   └── ui.js           # DOM rendering, user controls, structured commentary
│   └── assets/             # GIFs and images
├── poker_game_field.py     # Python implementation of the ζ-field model
├── zeta_field_advanced.py  # Advanced field with Heston/GARCH/Lévy components
├── zeta_field_python.py    # Core ζ-field Python implementation
├── zeta_field_r.R          # R implementation
├── ZetaField.java          # Java implementation
└── kuhn_poker_cfr.py       # Counterfactual Regret Minimization for Kuhn Poker
```

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/Zhandolia/broki.git
cd broki

# Serve the portfolio locally
cd portfolio
python3 -m http.server 8080

# Open http://localhost:8080
```

## 🃏 Game Features

- **Real-time ζ-field heatmap** — watch the action surface shift with every card
- **Model Insight panel** — structured commentary with headline, stats, and context
- **Live Analysis sidebar** — equity, pot odds, EV, confidence per hand
- **AI opponent** — makes decisions using the same field model
- **Suggestion bar** — real-time recommended action with confidence

## 🔬 Research Implementations

| File | Language | Description |
|------|----------|-------------|
| `poker_game_field.py` | Python | Full poker field model with visualization |
| `zeta_field_advanced.py` | Python | Heston SV, GARCH, Merton jump-diffusion |
| `ZetaField.java` | Java | Java port of the ζ-field model |
| `zeta_field_r.R` | R | R implementation for statistical analysis |
| `kuhn_poker_cfr.py` | Python | CFR algorithm on simplified Kuhn Poker |

## License

MIT
