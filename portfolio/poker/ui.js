/* ═══════════════════════════════════════════════════════════
   Poker UI — DOM Rendering & User Controls
   ═══════════════════════════════════════════════════════════ */
import { cardHTML, cardBackHTML } from './engine.js?v=7';
import { renderField, getRecommendation } from './field.js?v=7';
import { PokerGame } from './game.js?v=7';

const game = new PokerGame();

function $(id) { return document.getElementById(id); }

const STREET_LABELS = ['Pre-flop', 'Flop', 'Turn', 'River', 'Showdown'];
const ACTION_TONE = { RAISE: 'raise', BET: 'raise', CALL: 'call', CHECK: 'call', FOLD: 'fold' };

const money = n => { const r = Math.round(n); return (r < 0 ? '-$' : '$') + Math.abs(r); };
const signedMoney = n => (Math.round(n) >= 0 ? '+' : '') + money(n);
const pct = n => n.toFixed(0) + '%';

/* ── Cards ── */
function renderCards(el, cards, show) {
    el.innerHTML = cards.map(c => show ? cardHTML(c) : cardBackHTML()).join('');
}

function renderBoard(board) {
    let html = '';
    for (let i = 0; i < 5; i++) {
        html += i < board.length ? cardHTML(board[i]) : '<div class="card card-slot"></div>';
    }
    $('community-cards').innerHTML = html;
}

function renderLog(log) {
    $('action-log').innerHTML = log.slice(-3).map(l => `<li>${l}</li>`).join('');
}

/* ── Advisor: live decision ── */
function showAdvice(state) {
    const streetIdx = Math.max(0, state.streetIdx);
    const handStr = state.heroEval ? state.heroEval.handRank : 0;
    const aggr = state.toCall / Math.max(state.pot, 1);
    const rec = getRecommendation(state.equity, state.potOdds, streetIdx, state.toCall, state.pot);

    $('advice').dataset.tone = ACTION_TONE[rec.action];
    $('rec-action').textContent = rec.action === 'BET'
        ? `Bet $${rec.betSize}`
        : rec.action[0] + rec.action.slice(1).toLowerCase();
    $('rec-reason').textContent = rec.reason;

    const facing = state.toCall > 0;
    $('stat-equity').textContent = pct(state.equity);
    $('stat-potodds').textContent = facing ? pct(state.potOdds) : 'free';
    $('stat-ev').textContent = facing ? signedMoney(rec.ev) : '—';
    $('stat-ev').dataset.tone = !facing ? '' : rec.ev >= 0 ? 'good' : 'bad';

    renderField('field-canvas', state.equity, state.potOdds, streetIdx, aggr, handStr);
    return rec;
}

/* ── Advisor: hand result ── */
let profitBefore = 0;

function showResult(state) {
    const delta = state.heroProfit - profitBefore;
    const tone = state.winner === 'hero' ? 'call' : state.winner === 'opp' ? 'fold' : 'neutral';
    $('advice').dataset.tone = tone;
    $('rec-action').textContent = state.winner === 'hero' ? `You win ${money(delta)}`
        : state.winner === 'opp' ? `You lose ${money(-delta)}`
            : 'Split pot';
    $('rec-reason').textContent = state.message + '. Press Deal for the next hand.';
}

/* ── Full refresh from game state ── */
function updateUI(state) {
    const live = state.heroHole.length === 2 && !state.handOver;

    renderCards($('hero-cards'), state.heroHole, true);
    renderCards($('opp-cards'), state.oppHole, state.handOver);
    renderBoard(state.board);
    renderLog(state.log);

    $('hero-stack').textContent = money(state.heroStack);
    $('opp-stack').textContent = money(state.oppStack);
    $('pot-display').textContent = `Pot ${money(state.pot)}`;
    $('street-label').textContent = state.handOver ? 'Hand over' : STREET_LABELS[state.streetIdx] || '';
    $('hand-name').textContent = state.handName === '—' ? '' : state.handName;

    // Controls
    const canAct = live && state.isHeroTurn;
    $('btn-fold').disabled = !canAct || state.toCall <= 0;
    $('btn-call').disabled = !canAct;
    $('btn-raise').disabled = !canAct || state.heroStack <= state.toCall;
    $('btn-deal').disabled = live;

    let rec = null;
    if (live) rec = showAdvice(state);
    else if (state.handOver) showResult(state);

    if (canAct) {
        $('btn-call').textContent = state.toCall > 0 ? `Call $${state.toCall}` : 'Check';
        $('raise-label').textContent = state.toCall > 0 ? 'Raise +$' : 'Bet $';
        const slider = $('raise-slider');
        const max = Math.max(state.minRaise, state.heroStack - state.toCall);
        slider.min = state.minRaise;
        slider.max = max;
        // Pre-set to the model's sizing (60% pot), clamped to what's allowed
        const preset = rec && rec.betSize ? rec.betSize : Math.round(state.pot * 0.6);
        slider.value = Math.min(max, Math.max(state.minRaise, preset));
        $('raise-val').textContent = slider.value;
    }

    // Highlight whichever button the model suggests
    const suggested = rec ? { FOLD: 'btn-fold', CALL: 'btn-call', CHECK: 'btn-call', RAISE: 'btn-raise', BET: 'btn-raise' }[rec.action] : 'btn-deal';
    for (const id of ['btn-fold', 'btn-call', 'btn-raise', 'btn-deal']) {
        $(id).classList.toggle('suggested', id === suggested && !$(id).disabled);
    }

    // Session
    const finished = state.handsPlayed - (live ? 1 : 0);
    $('stat-hands').textContent = finished;
    $('stat-won').textContent = finished > 0 ? `${state.heroWins} (${pct(state.heroWins / finished * 100)})` : '—';
    $('stat-profit').textContent = signedMoney(state.heroProfit);
    $('stat-profit').dataset.tone = state.heroProfit > 0 ? 'good' : state.heroProfit < 0 ? 'bad' : '';
}

/* ── Wiring ── */
$('btn-deal').addEventListener('click', () => {
    profitBefore = game.heroProfit;
    updateUI(game.deal());
});
$('btn-fold').addEventListener('click', () => updateUI(game.heroFold()));
$('btn-call').addEventListener('click', () => updateUI(game.heroCall()));
$('btn-raise').addEventListener('click', () => updateUI(game.heroRaise(parseInt($('raise-slider').value, 10))));
$('raise-slider').addEventListener('input', e => { $('raise-val').textContent = e.target.value; });

// Open straight into a hand
profitBefore = game.heroProfit;
updateUI(game.deal());
