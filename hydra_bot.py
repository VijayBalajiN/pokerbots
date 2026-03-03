"""
Hydra — An adaptive, exploitative bot for Sneak Peek Hold'em.

Design philosophy
─────────────────
• Pre-flop  : Wider range than Argos.  Big Blind defends more from position.
• Auction   : Aggressive information strategy.
              Premium hands bid high to DENY the opponent information.
              Marginal hands bid to WIN information most urgently.
              Weak hands save chips with a minimal bid.
• Post-flop : Opponent model drives every decision.
              - fold_rate > 0.55 →  inflate equity with fold equity;
                                    bluff / semi-bluff / c-bet liberally.
              - fold_rate < 0.40 →  shrink effective equity; pure value only.
              - Always c-bet when we raised pre-flop and opponent is passive.
• Adaptation: Opponent model accumulates across all 1000 hands:
              tracks fold rate and aggression frequency.
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
        # Pre-build full deck as eval7.Card objects
        self.deck = {r + s: eval7.Card(r + s)
                     for r in '23456789TJQKA' for s in 'cdhs'}

        # ── Persistent opponent model (all 1000 hands) ────────────────────
        self._n_hands         = 0
        self._opp_folds       = 0   # times opponent folded before showdown
        self._opp_raised_cnt  = 0   # hands where opponent raised at least once
        self._opp_showdowns   = 0   # hands that reached showdown

    # ────────────────────────────── helpers ──────────────────────────────────

    def _chen(self, cards):
        """Classic Chen Formula hand-strength heuristic."""
        r1, s1 = cards[0][0], cards[0][1]
        r2, s2 = cards[1][0], cards[1][1]
        score = max(RANK_VAL[r1], RANK_VAL[r2])
        if r1 == r2:
            score = max(5.0, RANK_VAL[r1] * 2)
        if s1 == s2:
            score += 2
        gap = abs(RANK_ORD.index(r1) - RANK_ORD.index(r2))
        score -= (0, 1, 2, 4, 5)[min(gap, 4)]
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
        on    = 2 - len(opp_k)
        bn    = 5 - len(brd_c)
        need  = on + bn
        if need > len(avail):
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

    # ── Opponent model accessors ─────────────────────────────────────────────

    def _fold_rate(self):
        """Fraction of hands where the opponent folded."""
        return self._opp_folds / max(1, self._n_hands)

    def _aggression(self):
        """Fraction of hands where the opponent raised at least once."""
        return self._opp_raised_cnt / max(1, self._n_hands)

    # ── Action builders ──────────────────────────────────────────────────────

    def _sized_raise(self, cs, pot_frac):
        """
        Return ActionRaise(x) where x = min_raise + pot_frac * pot,
        clamped to legal raise_bounds.  Returns None if not allowed.
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
        self._score           = self._chen(cs.my_hand)
        self._raised_preflop  = False   # did WE raise pre-flop this hand?
        self._opp_raised_hand = False   # did OPPONENT apply pressure this hand?

    def on_hand_end(self, gi: GameInfo, cs: PokerState) -> None:
        """
        Update opponent model.

        Showdown detection: when the hand reaches showdown, runner.py
        processes the 'O' clause and sets opp_hands[active] to the
        opponent's full 2-card hand, so len(opp_revealed_cards) == 2.
        If opponent folded, that clause is never sent → len < 2.
        """
        self._n_hands += 1
        opp_rev = cs.opp_revealed_cards
        payoff  = cs.payoff

        if len(opp_rev) >= 2:
            self._opp_showdowns += 1        # reached showdown
        elif payoff > 0:
            self._opp_folds += 1            # we won, no showdown → they folded

        if self._opp_raised_hand:
            self._opp_raised_cnt += 1

    # ──────────────────────────── main entry ─────────────────────────────────

    def get_move(self, gi: GameInfo, cs: PokerState):
        # Emergency fallback to avoid time-out forfeit
        if gi.time_bank < 1.5:
            if cs.can_act(ActionCheck): return ActionCheck()
            if cs.can_act(ActionCall):  return ActionCall()
            return ActionFold()

        # Track opponent pressure within a hand
        if cs.cost_to_call > 0:
            self._opp_raised_hand = True

        if cs.street == 'auction':  return self._auction(cs)
        if cs.street == 'pre-flop': return self._preflop(cs)
        return self._postflop(cs)

    # ─────────────────────────── street logic ────────────────────────────────

    def _auction(self, cs):
        """
        Information-aggressive bidding strategy.

        • Premium  (score ≥ 12): bid ~40 % pot to DENY opponent information
          and often still win the peek. Our hand is already strong; we don't
          need the card, but we don't want them to have it.

        • Medium   (score 8–11): info is most valuable here. Bid using the
          info-value function (peaks at equity ≈ 50 %) with a floor to
          ensure we compete for the peek.

        • Weak     (score < 8): save chips; we're likely folding anyway.
        """
        s = self._score
        if s >= 12:
            bid = int(cs.pot * 0.40)
        elif s >= 8:
            eq  = self._mc_equity(cs.my_hand, cs.board, n=70)
            unc = 4.0 * eq * (1.0 - eq)            # info value ∈ [0,1]
            bid = int(cs.pot * (0.12 + 0.25 * unc))
        else:
            bid = int(cs.pot * 0.04)
        return ActionBid(max(0, min(bid, cs.my_chips)))

    def _preflop(self, cs):
        """
        Wider ranges than Argos, with positional awareness.

        BB (big blind) is in position post-flop on most streets, so we
        defend the BB with a wider calling threshold against steal-sized bets.

        Score ≥ 12 : premium  →  3-bet big or call
        Score ≥ 9  : strong   →  call up to 200
        Score ≥ 6  : medium   →  call up to 50 (80 from BB)
        Score ≥ 4  : specul.  →  call up to 25 (implied odds)
        """
        s, ctc, is_bb = self._score, cs.cost_to_call, cs.is_bb
        if ctc > 0:
            if s >= 12:
                r = self._sized_raise(cs, 1.5)
                return r if r else ActionCall()
            if s >= 9:
                return (self._call_or_fold(cs) if ctc <= 200 else
                        (ActionFold() if cs.can_act(ActionFold) else ActionCheck()))
            if s >= 6:
                thresh = 80 if is_bb else 50
                return (self._call_or_fold(cs) if ctc <= thresh else
                        (ActionFold() if cs.can_act(ActionFold) else ActionCheck()))
            if s >= 4 and ctc <= 25:
                return self._call_or_fold(cs)
            return ActionFold() if cs.can_act(ActionFold) else ActionCheck()
        else:
            if s >= 8:
                r = self._sized_raise(cs, 0.0)      # standard min-raise open
                if r:
                    self._raised_preflop = True
                    return r
            return ActionCheck() if cs.can_act(ActionCheck) else ActionCall()

    def _postflop(self, cs):
        """
        Exploitative post-flop driven by the opponent model.

        Effective equity = raw_mc_equity
                         + fold_bonus  (fold equity when opponent folds a lot)
                         − call_penalty (shrink bluffs when opponent is a station)

        fold_bonus   = fold_rate × 0.30
        call_penalty = max(0, 0.50 − fold_rate) × 0.12

        Bet sizing:
        • Raise  (fold_rate > 0.55): 85 % pot  — exploit fold tendency hard
        • Raise  (default):          70 % pot
        • C-bet bluff threshold:      eff_eq ≥ 0.45 + pre-flop raise + passive opp
        """
        opp    = cs.opp_revealed_cards or None
        n      = 100 if cs.street == 'flop' else 120
        eq     = self._mc_equity(cs.my_hand, cs.board, opp, n)
        fr     = self._fold_rate()
        ctc    = cs.cost_to_call
        pot    = max(1, cs.pot)

        # ── Effective equity incorporating opponent tendencies ────────────
        fold_bonus   = fr * 0.30
        call_penalty = max(0.0, 0.50 - fr) * 0.12
        eff_eq       = min(1.0, max(0.0, eq + fold_bonus - call_penalty))

        if ctc > 0:
            pot_odds = ctc / (pot + ctc)
            if eff_eq >= pot_odds + 0.16:           # strong edge: raise
                frac = 0.85 if fr > 0.55 else 0.70
                r = self._sized_raise(cs, frac)
                return r if r else self._call_or_fold(cs)
            if eff_eq >= pot_odds + 0.02:           # marginal edge: call
                return self._call_or_fold(cs)
            return ActionFold() if cs.can_act(ActionFold) else ActionCheck()
        else:
            if eff_eq >= 0.60:                      # value bet
                r = self._sized_raise(cs, 0.70)
                return r if r else self._check_or_call(cs)
            # C-bet / semi-bluff: only when we raised pre-flop AND opponent
            # folds frequently (confirmed by model)
            if eff_eq >= 0.45 and fr > 0.50 and self._raised_preflop:
                r = self._sized_raise(cs, 0.45)
                return r if r else self._check_or_call(cs)
            return ActionCheck() if cs.can_act(ActionCheck) else ActionCall()


if __name__ == '__main__':
    run_bot(Player(), parse_args())