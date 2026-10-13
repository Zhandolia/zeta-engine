# ζ-Field Poker AI Engine

> 8-component game-theoretic field mapping equity × pot-odds space. Monte Carlo simulation feeds a continuous decision surface — play against it live.

![ζ-Field Poker Engine](portfolio/assets/poker_game_analysis.gif)

## 🎯 Live Demo

The site has two pages:
- **Play** (`index.html`) — heads-up Texas Hold'em against the AI, with the model's suggestion, equity, pot odds, EV and the field heatmap next to the table
- **How it works** (`how-it-works.html`) — the pipeline, the decision rule, the field formula, and an interactive explorer

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
Monte Carlo equity (1,200 rollouts vs a random hand) + pot odds → decision rule (BET/RAISE/CHECK/CALL/FOLD)
                                                                → ζ-field heatmap (visualization)
```

## 📁 Project Structure

```
broki/
├── portfolio/              # Web application
│   ├── index.html          # Play page — the game
│   ├── how-it-works.html   # Model explanation + interactive field explorer
│   ├── style.css           # Styles for both pages
│   ├── poker/
│   │   ├── engine.js       # Card evaluation, Monte Carlo simulation, hand ranking
│   │   ├── field.js        # ζ-field computation, heatmap rendering, recommendation
│   │   ├── game.js         # Game state machine, AI opponent logic, action log
│   │   ├── ui.js           # Play page rendering and controls
│   │   └── explore.js      # How-it-works explorer
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
git clone https://github.com/Zhandolia/zeta-engine.git
cd zeta-engine

# Serve the portfolio locally
cd portfolio
python3 -m http.server 8080

# Open http://localhost:8080
```

## 🃏 Game Features

- **Model suggestion** — one recommended action with a plain-English reason; the matching button is highlighted
- **Live numbers** — equity, equity needed to call, and dollar EV of calling
- **Real-time ζ-field heatmap** — your spot plotted on the equity × pot-odds plane
- **Action log** — what you and the villain did this hand
- **AI opponent** — tight-aggressive thresholds on its own Monte Carlo equity (800 rollouts)

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
