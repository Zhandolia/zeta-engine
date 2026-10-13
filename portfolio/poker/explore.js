/* ═══════════════════════════════════════════════════════════
   Field Explorer — interactive Ψ(e,p) on the How-it-works page
   ═══════════════════════════════════════════════════════════ */
import { renderField, getRecommendation, fieldComponents } from './field.js?v=7';

function $(id) { return document.getElementById(id); }

const ACTION_TONE = { RAISE: 'raise', BET: 'raise', CALL: 'call', CHECK: 'call', FOLD: 'fold' };
const TERM_NAMES = { V: 'V', R: 'R', C: 'C', F: 'F', B: 'B', S: 'S', J: 'J', G: 'Γ' };
const POT = 100; // normalized pot before calling

function update() {
    const equity = +$('x-eq').value;
    const potOdds = +$('x-po').value;
    const streetIdx = +$('x-street').value;
    const handRank = +$('x-hand').value;

    $('o-eq').textContent = equity + '%';
    $('o-po').textContent = potOdds === 0 ? 'no bet' : potOdds + '%';

    // pot odds p = call / (pot + call)  →  call = p·pot / (1 − p)
    const po = potOdds / 100;
    const toCall = Math.round(po * POT / (1 - po));
    const aggr = toCall / POT;

    renderField('explore-canvas', equity, potOdds, streetIdx, aggr, handRank);

    const rec = getRecommendation(equity, potOdds, streetIdx, toCall, POT);
    $('x-advice').dataset.tone = ACTION_TONE[rec.action];
    $('x-action').textContent = rec.action[0] + rec.action.slice(1).toLowerCase();
    $('x-reason').textContent = toCall > 0 ? `${rec.reason} (Pot is $${POT}.)` : rec.reason;

    const comps = fieldComponents(equity, potOdds, streetIdx, aggr, handRank);
    const maxAbs = Math.max(...Object.values(comps).map(Math.abs), 1);
    $('x-bars').innerHTML = Object.entries(comps).map(([k, v]) => {
        const w = Math.abs(v) / maxAbs * 50;
        const style = v >= 0 ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%`;
        return `<div class="bar-row"><span class="bar-sym">${TERM_NAMES[k]}</span>` +
            `<span class="bar-track"><span class="bar ${v < 0 ? 'neg' : ''}" style="${style}"></span></span>` +
            `<span class="bar-val">${v >= 0 ? '+' : ''}${v.toFixed(1)}</span></div>`;
    }).join('');
}

for (const id of ['x-eq', 'x-po', 'x-street', 'x-hand']) {
    $(id).addEventListener('input', update);
}
update();
