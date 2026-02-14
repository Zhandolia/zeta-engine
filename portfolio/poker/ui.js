/* ═══════════════════════════════════════════════════════════
   Poker UI — DOM Rendering & User Controls
   ═══════════════════════════════════════════════════════════ */
import { cardHTML, cardBackHTML, cardStr } from './engine.js?v=6';
import { renderField, getRecommendation, getCommentary, getSuggestion } from './field.js?v=6';
import { PokerGame } from './game.js?v=6';

const game = new PokerGame();

function $(id) { return document.getElementById(id); }

/* ── Card rendering ── */
function renderCards(containerId, cards, show = true) {
    const el = $(containerId);
    if (!el) return;
    if (!cards || cards.length === 0) {
        el.innerHTML = '<span style="color:#555;font-family:var(--mono);font-size:0.8rem">—</span>';
        return;
    }
    el.innerHTML = cards.map(c => show ? cardHTML(c) : cardBackHTML()).join('');
}

function renderCommunity(board) {
    const el = $('community-cards');
    if (!el) return;
    let html = '';
    for (let i = 0; i < 5; i++) {
        html += i < board.length ? cardHTML(board[i]) : cardBackHTML();
    }
    el.innerHTML = html;
}

/* ── Commentary block (structured) ── */
function updateCommentary(lines) {
    const el = $('model-commentary');
    if (!el) return;

    const colorMap = { green: 'var(--green)', red: 'var(--red)', gold: 'var(--gold)' };

    el.innerHTML = `<div class="commentary-header">MODEL INSIGHT</div>` +
        lines.map(l => {
            const cls = `c-${l.type}`;
            const style = l.color ? ` style="color:${colorMap[l.color] || l.color}"` : '';
            return `<div class="${cls}"${style}>${l.text}</div>`;
        }).join('');
}

/* ── Update entire UI from game state ── */
function updateUI(state) {
    // Cards
    renderCards('hero-cards', state.heroHole, true);
    renderCards('opp-cards', state.oppHole, state.handOver);
    renderCommunity(state.board);

    // Stacks & pot
    $('hero-stack').textContent = `$${state.heroStack}`;
    $('opp-stack').textContent = `$${state.oppStack}`;
    $('pot-display').textContent = `POT: $${state.pot}`;

    // Controls
    const canAct = !state.handOver && state.isHeroTurn;
    $('btn-fold').disabled = !canAct;
    $('btn-call').disabled = !canAct;
    $('btn-raise').disabled = !canAct;
    $('btn-deal').disabled = !state.handOver && state.streetIdx >= 0;

    if (canAct) {
        $('btn-call').textContent = state.toCall > 0 ? `CALL $${state.toCall}` : 'CHECK';
        const slider = $('raise-slider');
        slider.min = state.minRaise;
        slider.max = Math.max(state.minRaise, state.heroStack - state.toCall);
        slider.value = Math.min(slider.value, slider.max);
        $('raise-val').textContent = slider.value;
    }

    // ── Active hand: field + recommendation + commentary ──
    if (state.heroHole.length === 2 && !state.handOver) {
        const aggr = state.toCall / Math.max(state.pot, 1);
        const handStr = state.heroEval ? state.heroEval.handRank : 0;
        const streetIdx = Math.max(0, state.streetIdx);

        renderField('field-canvas', state.equity, state.potOdds, streetIdx, aggr, handStr);

        const rec = getRecommendation(state.equity, state.potOdds, streetIdx, handStr);

        // Commentary
        const commentary = getCommentary(
            state.equity, state.potOdds, streetIdx, aggr, handStr,
            state.handName, rec.action
        );
        updateCommentary(commentary);

        // Suggestion line
        const sug = getSuggestion(state.equity, state.potOdds, streetIdx, handStr, state.toCall, state.pot);
        $('suggestion-icon').textContent = sug.icon;
        $('suggestion-text').textContent = sug.text;

        const colorMap = {
            RAISE: '#ff2020', CALL: '#00ff80', FOLD: '#4080ff',
            CHECK: '#00ff80', 'BLUFF RAISE': '#ff8000'
        };
        const recColor = colorMap[rec.action] || '#ffffff';

        $('rec-action').textContent = `▶ ${rec.action}`;
        $('rec-action').style.color = recColor;
        $('rec-details').textContent =
            `Equity: ${state.equity.toFixed(1)}%  |  Pot Odds: ${state.potOdds.toFixed(1)}%  |  EV: ${rec.ev >= 0 ? '+' : ''}${rec.ev.toFixed(2)}x  |  Confidence: ${rec.confidence}%`;
        $('recommendation').style.borderColor = recColor;

        // Field info
        $('fi-equity').textContent = state.equity.toFixed(1) + '%';
        $('fi-potodds').textContent = state.potOdds.toFixed(1) + '%';
        $('fi-ev').textContent = (rec.ev >= 0 ? '+' : '') + rec.ev.toFixed(2) + 'x';

        // ── Hand over: showdown / fold result ──
    } else if (state.handOver) {
        const winColor = state.winner === 'hero' ? '#00ff80' : state.winner === 'opp' ? '#ff4040' : '#ffd700';
        $('rec-action').textContent = state.winner === 'hero' ? '🏆 YOU WIN' : state.winner === 'opp' ? '❌ VILLAIN WINS' : '🤝 SPLIT POT';
        $('rec-action').style.color = winColor;
        $('rec-details').textContent = state.message;
        $('recommendation').style.borderColor = winColor;

        // Showdown commentary
        const endLines = [];
        if (state.winner === 'hero') {
            endLines.push('🏆 Field gradient confirmed: position was +EV');
            endLines.push(`Result: ${state.message}`);
        } else if (state.winner === 'opp') {
            endLines.push('❌ Villain held stronger field position');
            endLines.push(`Result: ${state.message}`);
        } else {
            endLines.push('🤝 Equal field values — split pot');
        }
        endLines.push('Click DEAL for next hand');
        updateCommentary(endLines);

        // Suggestion for showdown
        $('suggestion-icon').textContent = '🃏';
        $('suggestion-text').textContent = 'Press DEAL to play the next hand';
    }

    // ── Stats panel (always update) ──
    $('stat-street').textContent = state.street;
    $('stat-hand').textContent = state.handName;
    if (state.heroHole.length === 2) {
        const rec2 = getRecommendation(state.equity, state.potOdds, Math.max(0, state.streetIdx), state.heroEval ? state.heroEval.handRank : 0);
        $('stat-equity').textContent = state.equity.toFixed(1) + '%';
        $('stat-equity').style.color = state.equity > 55 ? '#00ff80' : state.equity < 35 ? '#ff4040' : '#e0e0e0';
        $('stat-potodds').textContent = state.potOdds.toFixed(1) + '%';
        $('stat-ev').textContent = (rec2.ev >= 0 ? '+' : '') + rec2.ev.toFixed(2) + 'x';
        $('stat-ev').style.color = rec2.ev >= 0 ? '#00ff80' : '#ff4040';
        $('stat-conf').textContent = rec2.confidence + '%';
    }
    if (state.handOver) {
        $('stat-street').textContent = 'SHOWDOWN';
    }

    // Session stats
    $('stat-hands').textContent = state.handsPlayed;
    $('stat-winrate').textContent = state.handsPlayed > 0
        ? (state.heroWins / state.handsPlayed * 100).toFixed(0) + '%' : '—';
    $('stat-profit').textContent = (state.heroProfit >= 0 ? '+$' : '-$') + Math.abs(state.heroProfit);
    $('stat-profit').style.color = state.heroProfit >= 0 ? '#00ff80' : '#ff4040';
}

/* ── Public API (exposed to window for onclick handlers) ── */
const pokerUI = {
    newHand() {
        const state = game.deal();
        updateUI(state);
    },

    fold() {
        const state = game.heroFold();
        updateUI(state);
    },

    call() {
        const state = game.heroCall();
        updateUI(state);
    },

    raise() {
        const amt = parseInt($('raise-slider').value);
        const state = game.heroRaise(amt);
        updateUI(state);
    }
};

window.pokerUI = pokerUI;

/* ── Slider live update ── */
const slider = $('raise-slider');
if (slider) {
    slider.addEventListener('input', () => {
        $('raise-val').textContent = slider.value;
    });
}
