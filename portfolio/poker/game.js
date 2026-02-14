/* ═══════════════════════════════════════════════════════════
   Poker Game — State Machine & AI Opponent
   ═══════════════════════════════════════════════════════════ */
import { newDeck, shuffle, evalHand, monteCarlo, potOdds, HAND_NAMES } from './engine.js?v=6';

const STREETS = ['PREFLOP', 'FLOP', 'TURN', 'RIVER', 'SHOWDOWN'];
const BIG_BLIND = 10;
const SMALL_BLIND = 5;
const STARTING_STACK = 500;

export class PokerGame {
    constructor() {
        this.reset();
        this.handsPlayed = 0;
        this.heroProfit = 0;
        this.heroWins = 0;
    }

    reset() {
        this.deck = shuffle(newDeck());
        this.heroHole = [];
        this.oppHole = [];
        this.board = [];
        this.pot = 0;
        this.heroStack = STARTING_STACK;
        this.oppStack = STARTING_STACK;
        this.street = -1; // -1 = not started
        this.heroBet = 0;   // amount bet THIS STREET
        this.oppBet = 0;    // amount bet THIS STREET
        this.toCall = 0;
        this.minRaise = BIG_BLIND;
        this.isHeroTurn = true;
        this.handOver = false;
        this.winner = null;
        this.message = '';
        this.deckIdx = 0;
        // Track whether the opponent has had a chance to act this round
        this.oppActed = false;
    }

    deal() {
        this.reset();
        this.deck = shuffle(newDeck());
        this.deckIdx = 0;

        // Deal hole cards
        this.heroHole = [this.deck[this.deckIdx++], this.deck[this.deckIdx++]];
        this.oppHole = [this.deck[this.deckIdx++], this.deck[this.deckIdx++]];

        // Post blinds (hero=SB, opp=BB for simplicity)
        this.heroStack -= SMALL_BLIND;
        this.oppStack -= BIG_BLIND;
        this.heroBet = SMALL_BLIND;
        this.oppBet = BIG_BLIND;
        this.pot = SMALL_BLIND + BIG_BLIND;
        this.toCall = BIG_BLIND - SMALL_BLIND;
        this.minRaise = BIG_BLIND;

        this.street = 0; // PREFLOP
        this.isHeroTurn = true;
        this.handOver = false;
        this.winner = null;
        this.oppActed = false; // BB hasn't acted yet (has option)
        this.message = 'Your action';
        this.handsPlayed++;

        return this.getState();
    }

    getState() {
        let handName = '—';
        let equity = 50;
        let po = 0;
        let heroEval = null;

        if (this.heroHole.length === 2) {
            if (this.board.length >= 3) {
                heroEval = evalHand(this.heroHole, this.board);
                handName = heroEval.name;
            } else {
                // Preflop description
                const r0 = this.heroHole[0] % 13, r1 = this.heroHole[1] % 13;
                const s0 = Math.floor(this.heroHole[0] / 13), s1 = Math.floor(this.heroHole[1] / 13);
                const ranks = 'AKQJT98765432';
                const hi = Math.max(r0, r1), lo = Math.min(r0, r1);
                handName = ranks[12 - hi] + ranks[12 - lo] + (s0 === s1 ? 's' : 'o');
            }
            equity = monteCarlo(this.heroHole, this.board, 1200);
            po = potOdds(this.toCall, this.pot);
        }

        return {
            street: this.street >= 0 ? STREETS[this.street] : '—',
            streetIdx: this.street,
            heroHole: this.heroHole,
            oppHole: this.oppHole,
            board: this.board,
            pot: this.pot,
            heroStack: this.heroStack,
            oppStack: this.oppStack,
            toCall: this.toCall,
            minRaise: this.minRaise,
            isHeroTurn: this.isHeroTurn,
            handOver: this.handOver,
            winner: this.winner,
            message: this.message,
            handName,
            equity,
            potOdds: po,
            handsPlayed: this.handsPlayed,
            heroProfit: this.heroProfit,
            heroWins: this.heroWins,
            heroEval,
        };
    }

    /* ── Check if both players are all-in → run out board ── */
    checkAllIn() {
        if (this.heroStack <= 0 && this.oppStack <= 0) {
            // Both all-in: deal remaining streets and go to showdown
            while (this.board.length < 5) {
                this.deckIdx++; // burn
                this.board.push(this.deck[this.deckIdx++]);
            }
            this.showdown();
            return true;
        }
        return false;
    }

    /* ── Hero Actions ── */

    heroFold() {
        if (this.handOver || !this.isHeroTurn) return this.getState();
        this.winner = 'opp';
        this.message = 'You folded — Villain wins';
        this.heroProfit -= (STARTING_STACK - this.heroStack);
        this.endHand();
        return this.getState();
    }

    heroCall() {
        if (this.handOver || !this.isHeroTurn) return this.getState();
        const call = Math.min(this.toCall, this.heroStack);
        this.heroStack -= call;
        this.pot += call;
        this.heroBet += call;
        this.toCall = 0;
        this.isHeroTurn = false;

        // Check all-in scenario
        if (this.checkAllIn()) return this.getState();

        if (this.oppActed) {
            // Opp already acted this round (e.g., opp bet/checked and hero is responding)
            // Since hero now matched, the round is over → advance street
            this.advanceStreet();
        } else {
            // Opp hasn't acted yet (e.g., preflop hero calls SB, BB still has option)
            this.aiAct();
        }
        return this.getState();
    }

    heroRaise(amount) {
        if (this.handOver || !this.isHeroTurn) return this.getState();
        const raise = Math.min(amount, this.heroStack);
        const totalCall = this.toCall + raise;
        this.heroStack -= totalCall;
        this.pot += totalCall;
        this.heroBet += totalCall;
        this.toCall = 0;
        this.minRaise = raise;

        this.isHeroTurn = false;
        // Opp must respond to the raise → they haven't acted on THIS bet yet
        this.oppActed = false;
        this.aiAct();
        return this.getState();
    }

    /* ── AI Opponent ── */

    aiAct() {
        if (this.handOver) return;

        // Mark that opp has now acted
        this.oppActed = true;

        // Compute AI equity
        const aiEquity = monteCarlo(this.oppHole, this.board, 800) / 100;
        const facingBet = this.heroBet - this.oppBet;
        const po = facingBet > 0 ? facingBet / (this.pot + facingBet) : 0;

        // Simple TAG (tight-aggressive) strategy
        let action;
        if (facingBet <= 0) {
            // Check or bet
            if (aiEquity > 0.60) {
                action = 'bet';
            } else if (aiEquity > 0.40 && Math.random() < 0.3) {
                action = 'bet';
            } else {
                action = 'check';
            }
        } else {
            // Facing a bet
            if (aiEquity > 0.55) {
                action = Math.random() < 0.4 ? 'raise' : 'call';
            } else if (aiEquity > po + 0.05) {
                action = 'call';
            } else if (Math.random() < 0.08) {
                action = 'call'; // occasional hero call
            } else {
                action = 'fold';
            }
        }

        switch (action) {
            case 'fold':
                this.winner = 'hero';
                this.message = 'Villain folds — You win!';
                this.heroWins++;
                this.heroProfit += this.pot - (STARTING_STACK - this.heroStack);
                this.endHand();
                break;

            case 'check':
                this.message = 'Villain checks';
                // Both checked or hero called and opp checks → advance
                this.advanceStreet();
                break;

            case 'call': {
                const call = Math.min(facingBet, this.oppStack);
                this.oppStack -= call;
                this.pot += call;
                this.oppBet += call;
                this.message = `Villain calls $${call}`;
                // Check all-in
                if (!this.checkAllIn()) {
                    // Both bets matched → advance
                    this.advanceStreet();
                }
                break;
            }

            case 'bet': {
                const betSize = Math.min(
                    Math.max(BIG_BLIND, Math.floor(this.pot * (0.5 + Math.random() * 0.3))),
                    this.oppStack
                );
                this.oppStack -= betSize;
                this.pot += betSize;
                this.oppBet += betSize;
                this.toCall = this.oppBet - this.heroBet;
                this.message = `Villain bets $${betSize}`;
                this.isHeroTurn = true;
                break;
            }

            case 'raise': {
                const total = facingBet + Math.min(
                    Math.max(BIG_BLIND, Math.floor(this.pot * 0.6)),
                    this.oppStack - facingBet
                );
                this.oppStack -= total;
                this.pot += total;
                this.oppBet += total;
                this.toCall = this.oppBet - this.heroBet;
                this.message = `Villain raises to $${this.oppBet}`;
                this.isHeroTurn = true;
                break;
            }
        }
    }

    /* ── Street Advancement ── */

    advanceStreet() {
        this.heroBet = 0;
        this.oppBet = 0;
        this.toCall = 0;
        this.oppActed = false;
        this.street++;

        if (this.street === 1) {
            // Flop
            this.deckIdx++; // burn
            this.board.push(this.deck[this.deckIdx++], this.deck[this.deckIdx++], this.deck[this.deckIdx++]);
            this.message = 'Flop dealt — your action';
            this.isHeroTurn = true;
        } else if (this.street === 2) {
            // Turn
            this.deckIdx++;
            this.board.push(this.deck[this.deckIdx++]);
            this.message = 'Turn dealt — your action';
            this.isHeroTurn = true;
        } else if (this.street === 3) {
            // River
            this.deckIdx++;
            this.board.push(this.deck[this.deckIdx++]);
            this.message = 'River dealt — your action';
            this.isHeroTurn = true;
        } else {
            // Showdown
            this.showdown();
        }
    }

    showdown() {
        this.street = 4;
        const heroEval = evalHand(this.heroHole, this.board);
        const oppEval = evalHand(this.oppHole, this.board);

        if (heroEval.score > oppEval.score) {
            this.winner = 'hero';
            this.message = `You win! ${heroEval.name} beats ${oppEval.name}`;
            this.heroWins++;
            this.heroProfit += this.pot - (STARTING_STACK - this.heroStack);
        } else if (heroEval.score < oppEval.score) {
            this.winner = 'opp';
            this.message = `Villain wins — ${oppEval.name} beats ${heroEval.name}`;
            this.heroProfit -= (STARTING_STACK - this.heroStack);
        } else {
            this.winner = 'tie';
            this.message = `Split pot! Both have ${heroEval.name}`;
        }
        this.endHand();
    }

    endHand() {
        this.handOver = true;
        this.isHeroTurn = false;
    }
}
