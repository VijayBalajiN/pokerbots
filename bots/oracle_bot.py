"""
Oracle — Evolved mc_with_ml for Sneak Peek Hold'em

Root-cause analysis of mc_with_ml's losses:
─────────────────────────────────────────────
1. FIXED AUCTION BIDS (248/251): mahoraga's Welford model learns these in ~10 hands
   and bids 252+ to outbid us. We then pay 251 for information we didn't get.
   Fix → Adaptive Welford auction: bid opp_max+margin for strong hands.

2. NO RANGE RESTRICTION WHEN OPP CARD KNOWN: mc_with_ml builds ~990 combos even
   when it knows one of opp's cards. The ML filter then slices this to top-X%.
   The card-containing combos are spread throughout the ranked list, so many are
   excluded. Fix → restrict range to ~44 combos containing the known card first,
   THEN apply ML percentile filter on those 44. This is a 20x precision boost.

3. CRUDE THREAT ASSESSMENT: mc_with_ml checks board ranks/suits manually. Fix →
   compute actual equity vs full revealed-constrained range; threat = max(0, 0.5-eq).

4. FIXED LEARNING RATE: can oscillate / blow up late game. Fix → decaying LR.

5. NO ANTI-ESCALATION: re-raises marginal hands into 3-bets → massive chip bleeds.
   Fix → only re-raise with safe_hand_types or equity > 0.65 + 0.20 edge margin.

6. HARD 1000-CHIP OVERBET DEFENCE: folds too late. Fix → graduated 400/600/1000.

7. BACKPROP BUG (single-card reveal): original uses index of median match, which
   is correct. But the y_true computation for single-card reveal with restricted
   range always yields 0.5. Keep unrestricted range for backprop when single card.
"""

import random
import itertools
import eval7
import math
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState, STARTING_STACK
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot


class Player(BaseBot):

    def __init__(self) -> None:
        # ── ML parameters with Bayesian priors ───────────────────────────────
        # Prior: theta=-1, b=1.5 → at pot=400 (x=1.0): z=0.5, y_hat≈0.62
        # Converges ~2x faster than zero-init (matches empirical avg pot range)
        self.theta = {'pre-flop': -1.0, 'flop': -1.0, 'turn': -1.0, 'river': -1.0}
        self.b     = {'pre-flop':  1.5, 'flop':  1.5, 'turn':  1.5, 'river':  1.5}
        self.learning_rate    = 0.01
        self.training_samples = 0   # only full 2-card reveals count
        self.current_hand_history = {}

        # ── Pot scaling (log tail prevents cap arbitrage) ─────────────────────
        self._POT_SCALE = 400.0
        self._POT_CAP   = 8.0

        # ── Adaptive auction: Welford online statistics ───────────────────────
        # Seeded with prior anchored near mc_with_ml / mahoraga bid range (~200).
        # std=50 → opp_max≈250 → premium bids start at 251 (correct vs mc_with_ml).
        self.opp_bid_n    = 1
        self.opp_bid_mean = 200.0
        self.opp_bid_M2   = 2500.0   # std = sqrt(M2/n) = 50 at n=1

        # Per-hand auction state
        self.my_auction_bid          = 0
        self.my_chips_before_auction = STARTING_STACK
        self.auction_resolved        = False

        # ── Per-hand state ────────────────────────────────────────────────────
        self.preflop_score = 0
        self._range_cache  = {}   # {street: [(combo, strength), ...]} — built once
        self._raised_pf    = False

        # ── Pre-allocated deck ────────────────────────────────────────────────
        self.full_deck = {r + s: eval7.Card(r + s)
                          for r in '23456789TJQKA' for s in 'cdhs'}
        self.preflop_percentiles = self._build_preflop_cache()

    # ─────────────────────────────────────────────────────────────────────────
    # Static helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_chen_score(self, cards):
        ranks = {
            'A': 10, 'K': 8, 'Q': 7, 'J': 6, 'T': 5,
            '9': 4.5, '8': 4, '7': 3.5, '6': 3, '5': 2.5,
            '4': 2, '3': 1.5, '2': 1
        }
        r1, s1 = cards[0][0], cards[0][1]
        r2, s2 = cards[1][0], cards[1][1]
        score = max(ranks[r1], ranks[r2])
        if r1 == r2:
            score = max(5, ranks[r1] * 2)
        if s1 == s2:
            score += 2
        gap = abs(ranks[r1] - ranks[r2])
        if   gap == 1: score -= 1
        elif gap == 2: score -= 2
        elif gap == 3: score -= 4
        elif gap >= 4: score -= 5
        return score

    def _build_preflop_cache(self):
        deck_strs = list(self.full_deck.keys())
        combos = list(itertools.combinations(deck_strs, 2))
        scored = [(frozenset([c1, c2]), self._get_chen_score([c1, c2]))
                  for c1, c2 in combos]
        scored.sort(key=lambda x: x[1], reverse=True)
        total = len(scored)
        return {fs: idx / total for idx, (fs, _) in enumerate(scored)}

    def _opp_bid_std(self):
        return math.sqrt(self.opp_bid_M2 / max(1, self.opp_bid_n))

    def _welford_update_auction(self, x):
        """One-step Welford update on the opponent bid model."""
        n     = self.opp_bid_n + 1
        delta = x - self.opp_bid_mean
        mean  = self.opp_bid_mean + delta / n
        M2    = self.opp_bid_M2 + delta * (x - mean)
        self.opp_bid_n, self.opp_bid_mean, self.opp_bid_M2 = n, mean, M2

    def _ml_predict(self, street, pot):
        """Return (y_hat, x_scaled) from the per-street logistic model."""
        x_raw = pot / self._POT_SCALE
        x_scaled = (x_raw if x_raw <= self._POT_CAP
                    else self._POT_CAP + math.log1p(x_raw - self._POT_CAP))
        z = self.theta[street] * x_scaled + self.b[street]
        y_hat = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))
        return max(0.02, min(1.0, y_hat)), x_scaled

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def on_hand_start(self, gi: GameInfo, cs: PokerState) -> None:
        self.preflop_score        = self._get_chen_score(cs.my_hand)
        self.my_auction_bid       = 0
        self.my_chips_before_auction = cs.my_chips
        self.auction_resolved     = False
        self.current_hand_history = {}
        self._range_cache         = {}
        self._raised_pf           = False

    def on_hand_end(self, gi: GameInfo, cs: PokerState) -> None:
        """Gradient descent backprop with decaying learning rate."""
        opp_cards = cs.opp_revealed_cards
        if not opp_cards:
            return
        if len(opp_cards) == 2:
            self.training_samples += 1

        for street, data in self.current_hand_history.items():
            if street == 'pre-flop':
                y_true = (self.preflop_percentiles.get(frozenset(opp_cards), 0.5)
                          if len(opp_cards) == 2 else 0.5)
            else:
                # BACKPROP BUG FIX: use the UNRESTRICTED range for backprop.
                # When we restricted to ~44 combos and only 1 card is revealed,
                # every combo matches → y_true always 0.5 → no learning.
                # We store the unrestricted range separately for this purpose.
                hs = data.get('backprop_evals') or data.get('cached_evals')
                if not hs:
                    continue
                if len(opp_cards) == 2:
                    opp_set = set(opp_cards)
                    y_true = 0.5
                    for idx, (hand, _) in enumerate(hs):
                        if {c.__str__() for c in hand} == opp_set:
                            y_true = idx / len(hs)
                            break
                else:
                    occ = opp_cards[0]
                    indices = [i for i, (h, _) in enumerate(hs)
                               if occ in [c.__str__() for c in h]]
                    y_true = ((sum(indices) / len(indices)) / len(hs)
                              if indices else 0.5)

            x_scaled = data['x_scaled']
            y_hat    = data['y_hat']
            error    = y_hat - y_true
            grad     = y_hat * (1.0 - y_hat)
            # Decaying LR: prevents wild swings late in the 1000-hand match
            lr = self.learning_rate / (1.0 + self.training_samples * 0.01)
            self.theta[street] -= lr * error * grad * x_scaled
            self.b[street]     -= lr * error * grad

    # ─────────────────────────────────────────────────────────────────────────
    # Auction resolution (called at first post-auction street)
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_auction(self, cs: PokerState):
        """
        Infer opp bid from chip delta and update Welford model.
        Winner pays LOSER's bid (second-price).
        chip_delta = my_chips_before_auction − current_my_chips
          → We won: chip_delta = opp's exact bid (we paid it)
          → We lost: chip_delta = 0 (we paid nothing)
        """
        if self.auction_resolved:
            return
        self.auction_resolved = True
        chip_delta = self.my_chips_before_auction - cs.my_chips
        we_won = bool(cs.opp_revealed_cards)
        if we_won and chip_delta > 0:
            self._welford_update_auction(float(chip_delta))
        else:
            # Lost: soft estimate anchored just above our bid
            soft = float(self.my_auction_bid) + max(5.0, self._opp_bid_std() * 0.5)
            self._welford_update_auction(min(soft, 600.0))

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry
    # ─────────────────────────────────────────────────────────────────────────

    def get_move(self, gi: GameInfo, cs: PokerState):
        # Emergency: burn no more CPU time if almost out
        if gi.time_bank < 1.5:
            return ActionCheck() if cs.can_act(ActionCheck) else ActionCall()

        # Resolve auction exactly once, at first post-auction action
        if cs.street in ('flop', 'turn', 'river'):
            self._resolve_auction(cs)

        if cs.street == 'auction':   return self._do_auction(cs)
        if cs.street == 'pre-flop':  return self._preflop(cs)
        return self._postflop(cs)

    # ─────────────────────────────────────────────────────────────────────────
    # Auction
    # ─────────────────────────────────────────────────────────────────────────

    def _do_auction(self, cs):
        """
        Adaptive bidding strategy:
        • Strong hand (score ≥ 9): bid opp_max + jitter to reliably WIN the peek.
          Cap at 450 to prevent runaway bidding wars (second-price: cost is opp bid,
          not ours — but opp's bid grows if they also have an adaptive model).
        • Medium hand (score 5–8): bid opp_mean − 1.
          (a) When opp is weak (~15 bid): WE WIN cheaply.
          (b) When opp is strong (~250 bid): OPP PAYS ~249 into the pot.
          Either outcome is +EV vs bidding near 0.
        • Weak hand: minimal bid — save chips, we'll fold post-flop anyway.
        """
        std     = self._opp_bid_std()
        opp_max = self.opp_bid_mean + std

        if self.preflop_score >= 9:
            bid = int(min(opp_max + 2 + random.randint(0, 5), 450))
        elif self.preflop_score >= 5:
            bid = max(1, int(self.opp_bid_mean) - 1)
        else:
            bid = 10

        # Snapshot chips BEFORE paying, so _resolve_auction can compute chip_delta
        self.my_chips_before_auction = cs.my_chips
        self.my_auction_bid = int(min(bid, cs.my_chips))
        return ActionBid(self.my_auction_bid)

    # ─────────────────────────────────────────────────────────────────────────
    # Pre-flop
    # ─────────────────────────────────────────────────────────────────────────

    def _preflop(self, cs):
        y_hat, x_scaled = self._ml_predict('pre-flop', float(cs.pot))
        self.current_hand_history['pre-flop'] = {
            'x_scaled': x_scaled, 'y_hat': y_hat,
            'cached_evals': None, 'backprop_evals': None
        }

        s, ctc = self.preflop_score, cs.cost_to_call
        if ctc > 0:
            if ctc <= 20:
                if s >= 9 and cs.can_act(ActionRaise):
                    lo, hi = cs.raise_bounds
                    target = max(lo, min(hi, lo + 40))
                    self._raised_pf = True
                    return ActionRaise(int(target))
                elif s >= 4:
                    return ActionCall() if cs.can_act(ActionCall) else ActionFold()
            else:
                if s >= 10:
                    if cs.pot > 400 or s < 14:
                        return ActionCall() if cs.can_act(ActionCall) else ActionFold()
                    if cs.can_act(ActionRaise):
                        lo, hi = cs.raise_bounds
                        target = max(lo, min(hi, lo + int(cs.pot * 0.5)))
                        self._raised_pf = True
                        return ActionRaise(int(target))
                    return ActionCall() if cs.can_act(ActionCall) else ActionFold()
                elif s >= 7 and ctc < 200:
                    return ActionCall() if cs.can_act(ActionCall) else ActionFold()
            return ActionFold() if cs.can_act(ActionFold) else ActionCheck()
        else:
            if s >= 9 and cs.can_act(ActionRaise):
                lo, hi = cs.raise_bounds
                self._raised_pf = True
                return ActionRaise(int(lo))
            return ActionCheck() if cs.can_act(ActionCheck) else ActionCall()

    # ─────────────────────────────────────────────────────────────────────────
    # Post-flop
    # ─────────────────────────────────────────────────────────────────────────

    def _postflop(self, cs):
        street = cs.street
        y_hat, x_scaled = self._ml_predict(street, float(cs.pot))
        predicted_percentile = y_hat

        # ── Build opponent range (once per street, cached) ────────────────────
        # KEY FIX: when opp card is known, restrict range to combos containing it.
        # dead = my_hand + board (NOT the revealed card — opp must have it).
        # This shrinks ~990 → ~44 combos, making ML filter far more precise.
        #
        # BACKPROP: store the FULL unrestricted range separately for on_hand_end.
        # With restricted range, single-card backprop always gives y_true=0.5
        # (all combos contain the known card), providing zero learning signal.
        # Full range still gives meaningful signal at showdown (2-card reveal).
        if street not in self._range_cache:
            dead = set(cs.my_hand + cs.board)
            avail = [k for k in self.full_deck if k not in dead]
            board_c = tuple(self.full_deck[k] for k in cs.board)
            revealed = cs.opp_revealed_cards

            # Always build full range (needed for backprop and threat level)
            full_combos = list(itertools.combinations(
                [self.full_deck[k] for k in avail], 2))
            full_hs = [(h, eval7.evaluate(list(h) + list(board_c)))
                       for h in full_combos]
            full_hs.sort(key=lambda x: x[1], reverse=True)

            if revealed and revealed[0] in self.full_deck and revealed[0] not in dead:
                # Restricted range: ~44 combos containing the known card
                rev = revealed[0]
                rest_combos = [(self.full_deck[rev], self.full_deck[c2])
                               for c2 in avail if c2 != rev]
                rest_hs = [(h, eval7.evaluate(list(h) + list(board_c)))
                           for h in rest_combos]
                rest_hs.sort(key=lambda x: x[1], reverse=True)
                play_hs = rest_hs   # equity computed on restricted range
            else:
                play_hs = full_hs   # no restriction

            self._range_cache[street] = {
                'play': play_hs,        # for equity decisions
                'full': full_hs         # for backprop
            }
            self.current_hand_history[street] = {
                'x_scaled':      x_scaled,
                'y_hat':         y_hat,
                'cached_evals':  play_hs,   # kept for compatibility
                'backprop_evals': full_hs
            }
        else:
            # Update ML record with latest pot value for backprop accuracy
            if street in self.current_hand_history:
                self.current_hand_history[street]['x_scaled'] = x_scaled
                self.current_hand_history[street]['y_hat']    = y_hat
            else:
                rc = self._range_cache[street]
                self.current_hand_history[street] = {
                    'x_scaled':      x_scaled,
                    'y_hat':         y_hat,
                    'cached_evals':  rc['play'],
                    'backprop_evals': rc['full']
                }

        play_hs = self._range_cache[street]['play']
        full_hs = self._range_cache[street]['full']

        if not play_hs:
            return ActionCheck() if cs.can_act(ActionCheck) else ActionCall()

        my_cards = [self.full_deck[k] for k in cs.my_hand]
        board_c  = [self.full_deck[k] for k in cs.board]

        # ── ML-filtered equity (core of mc_with_ml approach) ─────────────────
        num_keep = max(1, int(len(play_hs) * predicted_percentile))
        filtered = [(h, 1.0) for h, _ in play_hs[:num_keep]]
        num_mc   = min(400, max(100, len(filtered)))
        try:
            equity = eval7.py_hand_vs_range_monte_carlo(
                my_cards, filtered, board_c, num_mc)
        except Exception:
            equity = 0.5

        # ── Threat level: actual equity vs full revealed-constrained range ────
        # Better than mc_with_ml's crude board-rank/suit matching.
        threat_level = 0.0
        if cs.opp_revealed_cards and play_hs is not full_hs:
            # We restricted range → play_hs is card-constrained
            try:
                full_range = [(h, 1.0) for h, _ in play_hs]
                num_mc2 = min(200, max(50, len(full_range)))
                raw_eq = eval7.py_hand_vs_range_monte_carlo(
                    my_cards, full_range, board_c, num_mc2)
                threat_level = max(0.0, 0.5 - raw_eq)
            except Exception:
                threat_level = 0.0

        hand_value = eval7.evaluate(my_cards + board_c)
        hand_type  = eval7.handtype(hand_value)
        pot_odds   = cs.cost_to_call / max(1, cs.pot + cs.cost_to_call)
        safe_hands = {"Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"}

        # ── Raise safety gate ─────────────────────────────────────────────────
        can_raise_safely = cs.pot <= 800 or hand_type in safe_hands
        # Exploitation override: relax cap after sufficient training
        if self.training_samples >= 10 and equity > 0.85:
            if hand_type in {"Three of a Kind", "Two Pair"}:
                can_raise_safely = True

        # ── Decision: facing a bet ────────────────────────────────────────────
        if cs.cost_to_call > 0:
            ctc = cs.cost_to_call

            # Graduated overbet defence (vs mc_with_ml's single 1000-chip gate)
            if ctc > 400 and hand_type == "High Card" and equity < 0.45:
                return ActionFold() if cs.can_act(ActionFold) else ActionCheck()
            if ctc > 600 and hand_type == "Pair" and equity < 0.45:
                return ActionFold() if cs.can_act(ActionFold) else ActionCheck()
            if ctc > 1000 and equity < 0.35 and hand_type in {"High Card", "Pair"}:
                return ActionFold() if cs.can_act(ActionFold) else ActionCheck()

            # Always at least call with strong made hands (never fold a flush)
            if hand_type in safe_hands:
                if can_raise_safely and cs.can_act(ActionRaise) and equity > 0.65:
                    lo, hi = cs.raise_bounds
                    target = max(lo, min(hi, lo + int(cs.pot * 0.75)))
                    return ActionRaise(int(target))
                return ActionCall() if cs.can_act(ActionCall) else ActionFold()

            # Three of a Kind: protect from pessimistic ML filter folding
            if hand_type == "Three of a Kind" and equity > 0.30:
                return ActionCall() if cs.can_act(ActionCall) else ActionFold()

            required_equity = pot_odds + threat_level
            if equity > required_equity:
                # Anti-escalation: re-raise only with genuine nutted hands.
                # mc_with_ml's #1 chip leak: raise top-pair into 3-bet → fold → -800.
                can_reraise = False
                if can_raise_safely and cs.can_act(ActionRaise):
                    if hand_type in safe_hands and equity > 0.65:
                        can_reraise = True
                    elif hand_type == "Three of a Kind" and equity > 0.80:
                        can_reraise = True
                    elif equity > max(0.65, required_equity + 0.20):
                        can_reraise = True
                if can_reraise:
                    lo, hi = cs.raise_bounds
                    target = max(lo, min(hi, lo + int(cs.pot * 0.75)))
                    return ActionRaise(int(target))
                return ActionCall() if cs.can_act(ActionCall) else ActionFold()
            return ActionFold() if cs.can_act(ActionFold) else ActionCheck()

        # ── Decision: checking street ─────────────────────────────────────────
        else:
            if equity > (0.55 + threat_level):
                if can_raise_safely and cs.can_act(ActionRaise):
                    lo, hi = cs.raise_bounds
                    target = max(lo, min(hi, lo + int(cs.pot * 0.50)))
                    return ActionRaise(int(target))
            return ActionCheck() if cs.can_act(ActionCheck) else ActionCall()


if __name__ == '__main__':
    run_bot(Player(), parse_args())