#!/usr/bin/env python3
"""
Kuhn Poker CFR - Debug Version
================================
Testing terminal node payoffs explicitly.
"""

# Test the payoffs
def test_payoffs():
    """Manually verify payoff calculations."""
    
    print("\n" + "=" * 70)
    print("  Testing Kuhn Poker Payoffs")
    print("=" * 70)
    
    J, Q, K = 0, 1, 2
    
    test_cases = [
        # (p0_card, p1_card, history,  expected_payoff_p0, description)
        (K, Q, 'cc',   +1, "K vs Q, both check → K wins $1"),
        (J, Q, 'cc',   -1, "J vs Q, both check → J loses $1"),
        (K, Q, 'bb',   +2, "K vs Q, both bet → K wins $2"),
        (J, Q, 'bb',   -2, "J vs Q, both bet → J loses $2"),
        (K, Q, 'bc',   +1, "K bets, Q folds → K wins $1"),
        (J, Q, 'bc',   +1, "J bets, Q folds → J wins $1"),
        (K, Q, 'cbc',  -1, "K checks, Q bets, K folds → K loses $1"),
        (K, Q, 'cbb',  +2, "K checks, Q bets, K calls → K wins $2"),
        (J, Q, 'cbb',  -2, "J checks, Q bets, J calls → J loses $2"),
    ]
    
    all_pass = True
    
    for p0_card, p1_card, history, expected, desc in test_cases:
        actual = get_payoff_v2(p0_card, p1_card, history)
        
        status = "✓" if actual == expected else "✗"
        if actual != expected:
            all_pass = False
            
        print(f"  {status} {desc:45s} | Expected: {expected:+2d} | Got: {actual:+2d}")
    
    print("=" * 70)
    
    if all_pass:
        print("\n  ✓ All payoff tests passed!\n")
    else:
        print("\n  ✗ Some payoff tests failed!\n")
    
    return all_pass


def get_payoff_v2(p0_card, p1_card, history):
    """
    Correct payoff function for Kuhn Poker.
    
    Returns utility for Player 0.
    """
    # Check-check: showdown for $1
    if history == 'cc':
        return 1 if p0_card > p1_card else -1
    
    # Bet-fold: bettor wins $1
    elif history == 'bc':
        return 1  # P0 bet, P1 folded
    
    # Bet-call: showdown for $2
    elif history == 'bb':
        return 2 if p0_card > p1_card else -2
    
    # Check-bet-fold: bettor wins $1
    elif history == 'cbc':
        return -1  # P1 bet, P0 folded
    
    # Check-bet-call: showdown for $2
    elif history == 'cbb':
        return 2 if p0_card > p1_card else -2
    
    else:
        raise ValueError(f"Unknown terminal history: {history}")


if __name__ == "__main__":
    test_payoffs()
