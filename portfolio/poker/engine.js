/* ═══════════════════════════════════════════════════════════
   Poker Engine — Card Deck, Hand Evaluation, Equity
   ═══════════════════════════════════════════════════════════ */

const RANKS = '23456789TJQKA';
const SUITS = ['♠', '♥', '♦', '♣'];
const SUIT_CLASS = ['black', 'red', 'red', 'black'];

/* ── Helpers ── */
export function cardRank(c) { return c % 13; }
export function cardSuit(c) { return Math.floor(c / 13); }
export function cardStr(c) {
    return RANKS[cardRank(c)] + SUITS[cardSuit(c)];
}
export function cardHTML(c) {
    const r = RANKS[cardRank(c)];
    const s = SUITS[cardSuit(c)];
    const cls = SUIT_CLASS[cardSuit(c)];
    return `<div class="card card-face ${cls}"><span class="card-rank">${r}</span><span class="card-suit">${s}</span></div>`;
}
export function cardBackHTML() {
    return `<div class="card card-back"></div>`;
}

/* ── Deck ── */
export function newDeck() {
    const d = [];
    for (let i = 0; i < 52; i++) d.push(i);
    return d;
}
export function shuffle(deck) {
    const d = deck.slice();
    for (let i = d.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [d[i], d[j]] = [d[j], d[i]];
    }
    return d;
}

/* ── Hand Evaluation (0–8) ── */
const HAND_NAMES = [
    'High Card', 'One Pair', 'Two Pair', 'Three of a Kind',
    'Straight', 'Flush', 'Full House', 'Four of a Kind', 'Straight Flush'
];
export { HAND_NAMES };

export function evalHand(hole, board) {
    const all = hole.concat(board);
    const ranks = all.map(cardRank).sort((a, b) => b - a);
    const suits = all.map(cardSuit);

    // Count ranks
    const rc = new Map();
    for (const r of ranks) rc.set(r, (rc.get(r) || 0) + 1);
    const counts = [...rc.values()].sort((a, b) => b - a);

    // Flush?
    const sc = [0, 0, 0, 0];
    for (const s of suits) sc[s]++;
    const isFlush = sc.some(v => v >= 5);

    // Straight?
    const uniq = [...new Set(ranks)].sort((a, b) => a - b);
    let isStraight = false;
    for (let i = 0; i <= uniq.length - 5; i++) {
        if (uniq[i + 4] - uniq[i] === 4) isStraight = true;
    }
    // Wheel
    if ([12, 0, 1, 2, 3].every(r => rc.has(r))) isStraight = true;

    // Best 5-card tiebreaker score (simplified)
    const topRanks = [...rc.entries()]
        .sort((a, b) => (b[1] - a[1]) || (b[0] - a[0]))
        .map(e => e[0]);
    const kicker = topRanks.slice(0, 5);

    let handRank;
    if (isStraight && isFlush) handRank = 8;
    else if (counts[0] === 4) handRank = 7;
    else if (counts[0] === 3 && counts[1] >= 2) handRank = 6;
    else if (isFlush) handRank = 5;
    else if (isStraight) handRank = 4;
    else if (counts[0] === 3) handRank = 3;
    else if (counts[0] === 2 && counts[1] === 2) handRank = 2;
    else if (counts[0] === 2) handRank = 1;
    else handRank = 0;

    // Composite score for comparison
    let score = handRank * 1e10;
    for (let i = 0; i < kicker.length; i++) {
        score += kicker[i] * Math.pow(14, 4 - i);
    }
    return { handRank, name: HAND_NAMES[handRank], score, kicker };
}

/* ── Monte Carlo Equity (vs random hand) ── */
export function monteCarlo(hole, board, sims = 1500) {
    const used = new Set(hole.concat(board));
    const available = [];
    for (let i = 0; i < 52; i++) { if (!used.has(i)) available.push(i); }

    let wins = 0, ties = 0;
    for (let s = 0; s < sims; s++) {
        // Fisher-Yates partial shuffle
        const pool = available.slice();
        for (let i = pool.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [pool[i], pool[j]] = [pool[j], pool[i]];
        }
        const opp = [pool[0], pool[1]];
        let idx = 2;
        const fullBoard = board.concat(pool.slice(idx, idx + (5 - board.length)));

        const heroScore = evalHand(hole, fullBoard).score;
        const oppScore = evalHand(opp, fullBoard).score;

        if (heroScore > oppScore) wins++;
        else if (heroScore === oppScore) ties++;
    }
    return (wins + ties * 0.5) / sims * 100;
}

/* ── Pot Odds ── */
export function potOdds(toCall, pot) {
    if (toCall <= 0) return 0;
    return toCall / (pot + toCall) * 100;
}
