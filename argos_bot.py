"""
Argos — A precision equity-based bot for Sneak Peek Hold'em.

Design philosophy
─────────────────
• Pre-flop  : Tight-aggressive. Chen hand score gates every entry.
• Auction   : Bids using information-value theory.
              info_val = 4·eq·(1−eq)  →  peaks at equity ≈ 0.50.
              When our equity is near 50 %, knowing the opponent's card is
              most useful; we bid proportionally.
• Post-flop : Fast Monte Carlo equity vs strict pot odds. Never bluffs —
              every bet/raise requires a real equity edge.
• Adaptation: None — a near-fixed strategy that is hard to exploit.
"""

import random
import eval7
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

# ── Constants ────────────────────────────────────────────────────────────────

RANK_ORD = '23456789TJQKA'
RANK_VAL = {'2': 1, '3': 1.5, '4': 2, '5': 2.5, '6': 3, '7': 3.5,
            '8': 4, '9': 4.5, 'T': 5, 'J': 6, 'Q': 7, 'K': 8, 'A': 10}


class Player(BaseBot):

    def __init__(self):
        # Pre-build the full deck as eval7.Card objects once
        self.deck = {r + s: eval7.Card(r + s)
                     for r in '23456789TJQKA' for s in 'cdhs'}

    # ────────────────────────────── helpers ──────────────────────────────────

    def _chen(self, cards):
        """Classic Chen Formula hand-strength heuristic."""
        r1, s1 = cards[0][0], cards[0][1]
        r2, s2 = cards[1][0], cards[1][1]
        score = max(RANK_VAL[r1], RANK_VAL[r2])
        if r1 == r2:                                # pair
            score = max(5.0, RANK_VAL[r1] * 2)
        if s1 == s2:                                # suited bonus
            score += 2
        gap = abs(RANK_ORD.index(r1) - RANK_ORD.index(r2))
        score -= (0, 1, 2, 4, 5)[min(gap, 4)]      # gap penalty
        # bonus for gut-shot or open-ended connector below queen
        if 0 < gap <= 2 and min(RANK_VAL[r1], RANK_VAL[r2]) < RANK_VAL['Q']:
            score += 1
        return score

    def _mc_equity(self, my_s, board_s, opp_s=None, n=110):
        """
        Monte Carlo equity vs a uniformly random opponent range.

        my_s    : list of card strings for our hole cards
        board_s : list of card strings already on the board
        opp_s   : list of opponent's KNOWN card strings (from auction), or None
        n       : number of Monte Carlo samples
        """
        known = set(my_s + board_s + (opp_s or []))
        avail = [c for k, c in self.deck.items() if k not in known]
        my_c  = [self.deck[k] for k in my_s]
        brd_c = [self.deck[k] for k in board_s]
        opp_k = [self.deck[k] for k in (opp_s or [])]
        on    = 2 - len(opp_k)          # opponent cards still needed
        bn    = 5 - len(brd_c)          # board cards still needed
        need  = on + bn
        if need > len(avail):           # safety: shouldn't happen in practice
            return 0.5
        wins = ties = 0
        for _ in range(n):
            s    = random.sample(avail, need)
            opp  = opp_k + s[:on]
            brd  = brd_c + s[on:]
            my_v = eval7.evaluate(my_c + brd)
            op_v = eval7.evaluate(opp  + brd)
            if   my_v > op_v: wins += 1
            elif my_v == op_v: ties += 1
        return (wins + 0.5 * ties) / n

    def _sized_raise(self, cs, pot_frac):
        """
        Return ActionRaise(x) where x = min_raise + pot_frac * pot,
        clamped to legal raise_bounds.  Returns None if raise not allowed.
        """
        if not cs.can_act(ActionRaise):
            return None
        lo, hi = cs.raise_bounds
        amt = max(lo, min(hi, lo + int(cs.pot * pot_frac)))
        return ActionRaise(int(amt))

    def _call_or_fold(self, cs):
        return ActionCall() if cs.can_act(ActionCall) else ActionFold()

    def _check_or_call(self, cs):
        return ActionCheck() if cs.can_act(ActionCheck) else ActionCall()

    # ─────────────────────────── lifecycle ───────────────────────────────────

    def on_hand_start(self, gi: GameInfo, cs: PokerState) -> None:
        self._score = self._chen(cs.my_hand)    # compute once per hand

    def on_hand_end(self, gi: GameInfo, cs: PokerState) -> None:
        pass                                    # Argos does not adapt

    # ──────────────────────────── main entry ─────────────────────────────────

    def get_move(self, gi: GameInfo, cs: PokerState):
        # Emergency fallback: if time is critically short, check/call instantly
        if gi.time_bank < 1.5:
            if cs.can_act(ActionCheck): return ActionCheck()
            if cs.can_act(ActionCall):  return ActionCall()
            return ActionFold()

        if cs.street == 'auction':  return self._auction(cs)
        if cs.street == 'pre-flop': return self._preflop(cs)
        return self._postflop(cs)

    # ─────────────────────────── street logic ────────────────────────────────

    def _auction(self, cs):
        """
        Bid = pot × 0.22 × info_val
        info_val = 4·eq·(1−eq), maximised when we are a 50/50 coin-flip.
        This ensures we pay more for information exactly when it matters most.
        """
        eq       = self._mc_equity(cs.my_hand, cs.board, n=80)
        info_val = 4.0 * eq * (1.0 - eq)           # ∈ [0, 1]
        bid      = int(cs.pot * 0.22 * info_val)
        return ActionBid(max(0, min(bid, cs.my_chips)))

    def _preflop(self, cs):
        """
        Tight-aggressive preflop strategy gated on Chen score.

        Score ≥ 12  →  premium  (AA/KK/QQ/AKs)  :  3-bet or call any raise
        Score ≥ 9   →  strong   (JJ/TT/AQ/AK)   :  call up to 120
        Score ≥ 6   →  medium   (77-99 / broadway):  call up to 45
        Below 6     →  fold
        """
        s, ctc = self._score, cs.cost_to_call
        if ctc > 0:
            if s >= 12:
                r = self._sized_raise(cs, 1.0)
                return r if r else ActionCall()
            if s >= 9:
                return (self._call_or_fold(cs) if ctc <= 120 else
                        (ActionFold() if cs.can_act(ActionFold) else ActionCheck()))
            if s >= 6:
                return (self._call_or_fold(cs) if ctc <= 45 else
                        (ActionFold() if cs.can_act(ActionFold) else ActionCheck()))
            return ActionFold() if cs.can_act(ActionFold) else ActionCheck()
        else:
            # Free to check: open-raise strong hands, check the rest
            if s >= 9:
                r = self._sized_raise(cs, 0.0)  # minimum open raise (2 BB)
                return r if r else ActionCheck()
            return ActionCheck() if cs.can_act(ActionCheck) else ActionCall()

    def _postflop(self, cs):
        """
        Pure equity vs pot-odds.  Thresholds:
        • equity ≥ pot_odds + 0.18  →  raise  (75 % pot)
        • equity ≥ pot_odds + 0.03  →  call
        • else                      →  fold
        • facing no bet:
          equity ≥ 0.63             →  value bet  (60 % pot)
          else                      →  check
        """
        opp = cs.opp_revealed_cards or None
        n   = 100 if cs.street == 'flop' else 130
        eq  = self._mc_equity(cs.my_hand, cs.board, opp, n)
        ctc = cs.cost_to_call
        pot = max(1, cs.pot)

        if ctc > 0:
            pot_odds = ctc / (pot + ctc)
            if eq >= pot_odds + 0.18:
                r = self._sized_raise(cs, 0.75)
                return r if r else self._call_or_fold(cs)
            if eq >= pot_odds + 0.03:
                return self._call_or_fold(cs)
            return ActionFold() if cs.can_act(ActionFold) else ActionCheck()
        else:
            if eq >= 0.63:
                r = self._sized_raise(cs, 0.60)
                return r if r else self._check_or_call(cs)
            return self._check_or_call(cs)


if __name__ == '__main__':
    run_bot(Player(), parse_args())