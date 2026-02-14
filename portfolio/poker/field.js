/* ═══════════════════════════════════════════════════════════
   ζ-Field Renderer — 2D Heatmap + Commentary + Suggestion
   ═══════════════════════════════════════════════════════════ */

const GRID = 80;

/* Color stops: fold(blue) → call(cyan/green) → raise(red) */
const STOPS = [
    [0.00, 4, 18, 104], [0.15, 16, 60, 160],
    [0.30, 32, 140, 192], [0.45, 48, 192, 200],
    [0.60, 96, 220, 160], [0.75, 180, 240, 80],
    [0.90, 232, 180, 32], [1.00, 200, 20, 16]
];

function lerpColor(t) {
    t = Math.max(0, Math.min(1, t));
    for (let i = 1; i < STOPS.length; i++) {
        if (t <= STOPS[i][0]) {
            const lo = STOPS[i - 1], hi = STOPS[i];
            const f = (t - lo[0]) / (hi[0] - lo[0]);
            return [
                lo[1] + (hi[1] - lo[1]) * f | 0,
                lo[2] + (hi[2] - lo[2]) * f | 0,
                lo[3] + (hi[3] - lo[3]) * f | 0
            ];
        }
    }
    return [200, 20, 16];
}

/* ── 8-component field Ψ(e,p) ── */
function fieldValue(e, p, eq, po, streetIdx, aggr, handStr) {
    const en = e / 100, pn = p / 100;
    const eqn = eq / 100, pon = po / 100;

    const V = 65 * Math.exp(-3 * (en - eqn) ** 2 - 1.5 * (pn - pon) ** 2);
    const R = 50 * en ** 2.5 * Math.exp(-2 * (pn - 0.4) ** 2) * (1 + 0.3 * streetIdx);
    const C = 35 * Math.exp(-5 * (en - 0.45) ** 2) * (1 - pn) * Math.exp(-1 * (pn - pon) ** 2);
    const F = -20 * (1 - en) ** 2 * pn * Math.exp(-0.5 * en ** 2);
    const B = 18 * (1 - en) ** 2.5 * aggr * Math.exp(-3 * (pn - 0.6) ** 2);
    const S = 8 * streetIdx * en * (1 - en) * Math.sin(Math.PI * pn);
    const J = 15 * Math.tanh(8 * (en - pon)) * Math.exp(-2 * (pn - pon) ** 2);
    const G = 10 * handStr * en * Math.exp(-0.5 * (pn - 0.5) ** 2);

    return V + R + C + F + B + S + J + G;
}

/* ── Individual component values for commentary ── */
function fieldComponents(eq, po, streetIdx, aggr, handStr) {
    const en = eq / 100, pn = po / 100;

    return {
        V: 65 * Math.exp(-3 * 0 - 1.5 * 0),
        R: 50 * en ** 2.5 * Math.exp(-2 * (pn - 0.4) ** 2) * (1 + 0.3 * streetIdx),
        C: 35 * Math.exp(-5 * (en - 0.45) ** 2) * (1 - pn),
        F: -20 * (1 - en) ** 2 * pn * Math.exp(-0.5 * en ** 2),
        B: 18 * (1 - en) ** 2.5 * aggr * Math.exp(-3 * (pn - 0.6) ** 2),
        S: 8 * streetIdx * en * (1 - en) * Math.sin(Math.PI * pn),
        J: 15 * Math.tanh(8 * (en - pn)),
        G: 10 * (handStr / 8) * en * Math.exp(-0.5 * (pn - 0.5) ** 2),
    };
}

/* ── Render 2D heatmap to canvas ── */
export function renderField(canvasId, equity, potOdds, streetIdx, aggr, handStr) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const cellW = W / GRID, cellH = H / GRID;

    // Compute field grid
    const data = new Float64Array(GRID * GRID);
    let mn = Infinity, mx = -Infinity;
    for (let j = 0; j < GRID; j++) {
        for (let i = 0; i < GRID; i++) {
            const e = (i / (GRID - 1)) * 100;
            const p = (1 - j / (GRID - 1)) * 100;
            const v = fieldValue(e, p, equity, potOdds, streetIdx, aggr, handStr / 8);
            data[j * GRID + i] = v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
    }

    // Draw cells
    const range = mx - mn || 1;
    const img = ctx.createImageData(W, H);
    for (let j = 0; j < GRID; j++) {
        for (let i = 0; i < GRID; i++) {
            const t = (data[j * GRID + i] - mn) / range;
            const [r, g, b] = lerpColor(t);
            const x0 = Math.floor(i * cellW), y0 = Math.floor(j * cellH);
            const x1 = Math.floor((i + 1) * cellW), y1 = Math.floor((j + 1) * cellH);
            for (let py = y0; py < y1 && py < H; py++) {
                for (let px = x0; px < x1 && px < W; px++) {
                    const idx = (py * W + px) * 4;
                    img.data[idx] = r;
                    img.data[idx + 1] = g;
                    img.data[idx + 2] = b;
                    img.data[idx + 3] = 255;
                }
            }
        }
    }
    ctx.putImageData(img, 0, 0);

    // Break-even diagonal
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.moveTo(0, H);
    ctx.lineTo(W, 0);
    ctx.strokeStyle = 'rgba(255,0,255,0.35)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.setLineDash([]);

    // Decision point marker (glowing dot)
    const dotX = (equity / 100) * W;
    const dotY = (1 - potOdds / 100) * H;
    ctx.beginPath();
    ctx.arc(dotX, dotY, 8, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(dotX, dotY, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#00ff80';
    ctx.shadowColor = '#00ff80';
    ctx.shadowBlur = 16;
    ctx.fill();
    ctx.shadowBlur = 0;

    // Axis labels
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillStyle = '#808090';
    ctx.textAlign = 'center';
    ctx.fillText('EQUITY →', W / 2, H - 4);
    ctx.save();
    ctx.translate(12, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('POT ODDS →', 0, 0);
    ctx.restore();

    // Title
    ctx.font = 'bold 11px JetBrains Mono, monospace';
    ctx.fillStyle = '#a0a8b8';
    ctx.textAlign = 'center';
    ctx.fillText('Ψ(e,p) ACTION FIELD', W / 2, 14);
}

/* ── Generate model commentary (structured) ── */
export function getCommentary(equity, potOdds, streetIdx, aggr, handStr, handName, action) {
    const comps = fieldComponents(equity, potOdds, streetIdx, aggr, handStr);
    const lines = [];

    const entries = Object.entries(comps);
    entries.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const top = entries[0];

    const compNames = {
        V: 'V_Equity', R: 'R_Raise', C: 'C_Call', F: 'F_Fold',
        B: 'B_Bluff', S: 'S_Street', J: 'J_Jump', G: 'Γ_Str'
    };

    // Headline: current equity read
    if (equity > 65) {
        lines.push({ type: 'headline', text: '⚡ Strong position — field peaks in raise zone', color: 'green' });
    } else if (equity > 45) {
        lines.push({ type: 'headline', text: '📊 Mixed signal — moderate equity region', color: 'gold' });
    } else if (equity > 30) {
        lines.push({ type: 'headline', text: '⚠️ Marginal — field tilts toward defense', color: 'gold' });
    } else {
        lines.push({ type: 'headline', text: '🔻 Weak — field drops into fold territory', color: 'red' });
    }

    // Dominant driver
    lines.push({ type: 'stat', text: `Dominant: ${compNames[top[0]]} = ${top[1].toFixed(1)}` });

    // Pot odds analysis
    if (potOdds > 0) {
        const needed = potOdds;
        if (equity > needed + 10) {
            lines.push({ type: 'stat', text: `Eq ${equity.toFixed(0)}% > PO ${needed.toFixed(0)}% — +EV`, color: 'green' });
        } else if (equity > needed - 5) {
            lines.push({ type: 'stat', text: `Eq ${equity.toFixed(0)}% ≈ PO ${needed.toFixed(0)}% — marginal`, color: 'gold' });
        } else {
            lines.push({ type: 'stat', text: `Eq ${equity.toFixed(0)}% < PO ${needed.toFixed(0)}% — -EV`, color: 'red' });
        }
    } else {
        lines.push({ type: 'stat', text: 'Free to act — no bet to face' });
    }

    // Component breakdown
    const active = entries.filter(e => Math.abs(e[1]) > 3).slice(0, 3);
    const breakdownStr = active.map(e => `${e[0]}=${e[1] > 0 ? '+' : ''}${e[1].toFixed(0)}`).join(' ');
    lines.push({ type: 'detail', text: `Components: ${breakdownStr}` });

    // Street context
    const streetNames = [
        'Pre-flop: wide ranges, high uncertainty',
        'Flop: range narrows, equity crystallizes',
        'Turn: critical street, pot commitment rises',
        'River: final decision, maximum information'
    ];
    if (streetIdx >= 0 && streetIdx < 4) {
        lines.push({ type: 'context', text: streetNames[streetIdx] });
    }

    return lines;
}

/* ── Get suggestion line for user ── */
export function getSuggestion(equity, potOdds, streetIdx, handStr, toCall, pot) {
    const eq = equity / 100;
    const po = potOdds / 100;

    if (toCall <= 0) {
        // No bet to face
        if (eq > 0.65 + 0.04 * streetIdx) {
            const betSize = Math.max(10, Math.round(pot * 0.6));
            return { text: `Bet $${betSize} for value — your hand is strong enough to build the pot`, icon: '💰' };
        } else if (eq > 0.45) {
            return { text: 'Check to control pot size — hand is decent but vulnerable', icon: '🎯' };
        } else {
            return { text: 'Check and see a free card — not worth building the pot here', icon: '👀' };
        }
    } else {
        // Facing a bet
        if (eq > 0.70) {
            return { text: `Raise! You are well ahead — punish villain for betting into strength`, icon: '🚀' };
        } else if (eq > po + 0.08) {
            return { text: `Call — you have the equity to justify ($${toCall} into $${pot} pot)`, icon: '✅' };
        } else if (eq > po - 0.03) {
            return { text: `Close decision — calling is borderline, consider villain tendencies`, icon: '⚖️' };
        } else {
            return { text: `Fold — the price is too high relative to your equity`, icon: '🛑' };
        }
    }
}

/* ── Get recommendation ── */
export function getRecommendation(equity, potOdds, streetIdx, handStr) {
    const eq = equity / 100;
    const po = potOdds / 100;
    const evCall = eq * (1 + 1 / Math.max(po, 0.01)) - 1;

    let action, confidence, ev;
    if (eq > 0.65 + 0.05 * streetIdx) {
        action = 'RAISE';
        confidence = Math.min(99, 60 + 40 * eq | 0);
        ev = evCall * 1.5;
    } else if (eq > po - 0.05 && eq > 0.30) {
        action = 'CALL';
        confidence = Math.min(95, 40 + 50 * eq | 0);
        ev = evCall;
    } else if (potOdds === 0) {
        action = 'CHECK';
        confidence = 70;
        ev = 0;
    } else {
        action = 'FOLD';
        confidence = Math.min(95, 50 + 40 * (1 - eq) | 0);
        ev = 0;
    }
    return { action, confidence, ev };
}
