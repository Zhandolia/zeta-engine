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
export function fieldComponents(eq, po, streetIdx, aggr, handStr) {
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
    ctx.font = '12px JetBrains Mono, monospace';
    ctx.fillStyle = 'rgba(255,255,255,0.75)';
    ctx.shadowColor = 'rgba(0,0,0,0.8)';
    ctx.shadowBlur = 4;
    ctx.textAlign = 'center';
    ctx.fillText('EQUITY →', W / 2, H - 8);
    ctx.save();
    ctx.translate(16, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('POT ODDS →', 0, 0);
    ctx.restore();
    ctx.shadowBlur = 0;
}

/* ── Recommendation: equity vs. price of calling ──
   Value-bet bar is 65% equity, +5 points per street.
   Facing a bet: call within 5 points of break-even, never below 30%.
   ev = dollar EV of calling: equity × (pot + call) − call. */
const STREET_PHRASE = ['pre-flop', 'on the flop', 'on the turn', 'on the river'];

export function valueBar(streetIdx) {
    return 65 + 5 * streetIdx;
}

export function getRecommendation(equity, potOdds, streetIdx, toCall, pot) {
    const facingBet = toCall > 0;
    const ev = facingBet ? (equity / 100) * (pot + toCall) - toCall : 0;
    const bar = valueBar(streetIdx);
    const e = Math.round(equity), need = Math.round(potOdds);
    const where = STREET_PHRASE[streetIdx] || '';

    if (equity > bar) {
        if (!facingBet) {
            const betSize = Math.max(10, Math.round(pot * 0.6));
            return {
                action: 'BET', betSize, ev,
                reason: `${e}% equity clears the ${bar}% value bar ${where}. Bet about 60% of the pot.`
            };
        }
        return {
            action: 'RAISE', ev,
            reason: `${e}% equity is far above the ${need}% you need to call, and clears the ${bar}% value bar. Raise.`
        };
    }
    if (!facingBet) {
        return {
            action: 'CHECK', ev,
            reason: `${e}% equity is under the ${bar}% bar for betting ${where}. ` +
                (streetIdx >= 3 ? 'Check and go to showdown for free.' : 'Check and see the next card for free.')
        };
    }
    if (equity > potOdds - 5 && equity > 30) {
        return {
            action: 'CALL', ev,
            reason: equity >= potOdds
                ? `Calling $${toCall} needs ${need}% equity and you have ${e}%.`
                : `Close spot: calling $${toCall} needs ${need}% and you have ${e}%. The model still calls within 5 points of break-even.`
        };
    }
    return {
        action: 'FOLD', ev,
        reason: equity <= 30 && equity > potOdds - 5
            ? `${e}% equity is below the model's 30% floor for calling a bet.`
            : `Calling $${toCall} needs ${need}% equity and you have only ${e}%.`
    };
}
