"""
mc_with_ml v3 — surgical upgrade of the proven mc_with_ml base.

Base: original mc_with_ml (W-L 17-8, +266k) with ML structure untouched.
Changes: exactly 4 targeted grafts from mahoraga (the 24-1, +322k champion).

═══════════════════════════════════════════════════════════════════════════
WHY v2 FAILED (−168k, 8-17):
═══════════════════════════════════════════════════════════════════════════
  v2 added `revealed` to dead_strings. Since the revealed card was then
  excluded from the deck, no combo in hand_strengths contained it.
  Filtering `hand_strengths` by `rev_card_obj in hand` → empty list.
  empty filtered_range → MC exception → equity = 0.0 → always fold.
  Result: folded strong hands every time we won the auction. Disaster.

═══════════════════════════════════════════════════════════════════════════
ROOT CAUSES OF mc_with_ml LOSING TO mahoraga (5-0 every match):
═══════════════════════════════════════════════════════════════════════════

1. AUCTION (biggest edge):
   mc_with_ml bids 15 (weak), 248/251 (other).
   Mahoraga bids opp_bid_mean-1 (~199) for all non-premium hands.
   When mc_with_ml bids 15 and mahoraga bids 199:
     → mahoraga wins auction, pays only 15 chips, sees mc_with_ml's card.
   When mc_with_ml bids 251 (good hand) and mahoraga bids 252:
     → mahoraga still wins, pays 251 into pot.
   Result: mahoraga wins EVERY auction, gaining total information dominance.

2. ANTI-ESCALATION (chip bleeder):
   mc_with_ml re-raises whenever equity > required + 0.20, ANY hand type.
   In raise wars with two pair vs a set, mc_with_ml escalates and loses
   2000-4000 chips per hand. Mahoraga only re-raises with straight+/3oak.

3. STRONG HAND PROTECTION (gifted pots):
   mc_with_ml can fold a flush if ML model predicts tight range →
   equity (vs tight range) < required → folds → gifts the pot.
   Mahoraga: if hand_type in safe_hand_types → always at least call.

4. GRADUATED OVERBET DEFENSE (slow bleed):
   mc_with_ml only folds HC/pair at >1000 chip bets. Mahoraga folds:
     • HC at 400 chips (equity < 0.45)
     • Pair at 600 chips (equity < 0.45)
   This saves dozens of chips per hand over 1000 rounds.

═══════════════════════════════════════════════════════════════════════════
FIXES (surgical, nothing else changed):
═══════════════════════════════════════════════════════════════════════════
  Fix 1: Welford adaptive auction model (from mahoraga verbatim)
  Fix 2: Iron-clad strong hand protection + anti-escalation
  Fix 3: Graduated overbet defense (400/600/1000 thresholds)
  Fix 4: Decaying learning rate + unbiased y_true (mean not median)

UNCHANGED: ML structure, x_scaled formula, range-building logic,
           pre-flop logic, dead_strings (no revealed), all constants.
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
        self.hands_played = 0

        # ML Parameters — identical to original mc_with_ml
        self.theta = {'pre-flop': 0.0, 'flop': 0.0, 'turn': 0.0, 'river': 0.0}
        self.b     = {'pre-flop': 0.0, 'flop': 0.0, 'turn': 0.0, 'river': 0.0}
        self.learning_rate = 0.01
        self.training_samples = 0   # counts full 2-card reveals for decaying LR

        self.current_hand_history = {}

        # ── FIX 1: Welford adaptive auction model ────────────────────────────
        # Seeded at mean=200 (near typical competitive bid) so the model
        # predicts opp_bid_max ~ 250 from hand 1. Converges in ~5 rounds.
        # (Original mc_with_ml: static bids, loses auction vs mahoraga every hand)
        self.opp_bid_n    = 1
        self.opp_bid_mean = 200.0
        self.opp_bid_M2   = 2500.0   # std = sqrt(2500/1) = 50; pred_max = 250

        # Per-hand auction state
        self.my_chips_before_auction = STARTING_STACK
        self.my_auction_bid          = 0
        self.auction_resolved        = False

        # Pre-allocated deck — identical to original
        all_ranks = '23456789TJQKA'
        all_suits = 'cdhs'
        self.full_deck = {r + s: eval7.Card(r + s) for r in all_ranks for s in all_suits}
        self.preflop_percentiles = self._build_preflop_cache()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_chen_score(self, cards):
        ranks = {'A': 10, 'K': 8, 'Q': 7, 'J': 6, 'T': 5, '9': 4.5, '8': 4,
                 '7': 3.5, '6': 3, '5': 2.5, '4': 2, '3': 1.5, '2': 1}
        rank1, suit1 = cards[0][0], cards[0][1]
        rank2, suit2 = cards[1][0], cards[1][1]
        score = max(ranks[rank1], ranks[rank2])
        if rank1 == rank2: score = max(5, ranks[rank1] * 2)
        if suit1 == suit2: score += 2
        gap = abs(ranks[rank1] - ranks[rank2])
        if   gap == 1: score -= 1
        elif gap == 2: score -= 2
        elif gap == 3: score -= 4
        elif gap >= 4: score -= 5
        return score

    def _build_preflop_cache(self):
        deck_strs = list(self.full_deck.keys())
        scored = [(frozenset([c1, c2]), self._get_chen_score([c1, c2]))
                  for c1, c2 in itertools.combinations(deck_strs, 2)]
        scored.sort(key=lambda x: x[1], reverse=True)
        total = len(scored)
        return {combo: idx / total for idx, (combo, _) in enumerate(scored)}

    def _welford_update(self, n, mean, M2, x):
        n    += 1
        delta = x - mean
        mean += delta / n
        M2   += delta * (x - mean)
        return n, mean, M2

    def _welford_std(self, n, M2):
        return math.sqrt(M2 / max(n, 1))

    def _resolve_auction(self, current_state: PokerState):
        """
        Called once at the first post-auction street.
        Infers opponent's actual bid from chip delta and updates Welford model.

        Auction rule: winner pays the loser's bid.
          → We won:  our chips dropped by opp_bid → opp_bid = chips_before − chips_now
          → We lost: chips unchanged; opponent bid > our bid
        """
        if self.auction_resolved:
            return
        self.auction_resolved = True

        chip_delta = self.my_chips_before_auction - current_state.my_chips
        we_won = bool(current_state.opp_revealed_cards)

        if we_won and chip_delta > 0:
            # Exact observation: we paid their bid, so their bid = chip_delta
            self.opp_bid_n, self.opp_bid_mean, self.opp_bid_M2 = self._welford_update(
                self.opp_bid_n, self.opp_bid_mean, self.opp_bid_M2, float(chip_delta))
        else:
            # Lost: their bid was above ours. Anchor soft estimate just above our bid.
            opp_bid_std = self._welford_std(self.opp_bid_n, self.opp_bid_M2)
            soft = self.my_auction_bid + max(5.0, opp_bid_std * 0.5)
            soft = min(soft, 500.0)
            self.opp_bid_n, self.opp_bid_mean, self.opp_bid_M2 = self._welford_update(
                self.opp_bid_n, self.opp_bid_mean, self.opp_bid_M2, soft)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.hands_played += 1
        self.preflop_score = self._get_chen_score(current_state.my_hand)
        self.current_hand_history = {}
        self.my_chips_before_auction = current_state.my_chips
        self.my_auction_bid          = 0
        self.auction_resolved        = False

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        """Backpropagation with FIX 4: decaying LR + unbiased y_true (mean not median)."""
        opp_cards = current_state.opp_revealed_cards
        if not opp_cards:
            return

        if len(opp_cards) == 2:
            self.training_samples += 1

        for street, state_data in self.current_hand_history.items():
            if street == 'pre-flop':
                if len(opp_cards) == 2:
                    y_true = self.preflop_percentiles.get(frozenset(opp_cards), 0.5)
                else:
                    y_true = 0.5
            else:
                hand_strengths = state_data['cached_evals']
                if not hand_strengths:
                    continue

                if len(opp_cards) == 2:
                    opp_set = set(opp_cards)
                    y_true  = 0.5
                    for idx, (hand, _) in enumerate(hand_strengths):
                        if set(c.__str__() for c in hand) == opp_set:
                            y_true = idx / len(hand_strengths)
                            break
                else:
                    # FIX: unbiased expectation = mean index (was biased median)
                    opp_card_str = opp_cards[0]
                    indices = [idx for idx, (hand, _) in enumerate(hand_strengths)
                               if opp_card_str in [c.__str__() for c in hand]]
                    y_true = ((sum(indices) / len(indices)) / len(hand_strengths)
                              if indices else 0.5)

            x_scaled = state_data['x_scaled']
            y_hat    = state_data['y_hat']
            error    = y_hat - y_true
            grad_sig = y_hat * (1.0 - y_hat)

            # FIX: decaying learning rate — prevents over-fitting late in game
            lr = self.learning_rate / (1.0 + self.training_samples * 0.01)
            self.theta[street] -= lr * (error * grad_sig * x_scaled)
            self.b[street]     -= lr * (error * grad_sig)

    # ── main entry ────────────────────────────────────────────────────────────

    def get_move(self, game_info: GameInfo, current_state: PokerState):

        # Resolve auction chip delta on first post-auction street
        if current_state.street in ('flop', 'turn', 'river'):
            self._resolve_auction(current_state)

        # ── FIX 1: ADAPTIVE WELFORD AUCTION ──────────────────────────────────
        if current_state.street == 'auction':
            opp_std     = self._welford_std(self.opp_bid_n, self.opp_bid_M2)
            opp_pred_max = self.opp_bid_mean + opp_std

            if self.preflop_score >= 10:
                # Premium hand: WANT to win → bid just above predicted max
                # Small random jitter prevents exact reverse-engineering of our hand.
                bid = int(opp_pred_max + 1 + random.randint(0, 5))
            else:
                # All other hands: bid opp_bid_mean − 1.
                # Why this is better than bidding 15/248/251:
                #   (a) When opp bids LOW (weak hand, e.g. 15 chips):
                #       mean−1 = ~199 > 15 → WE WIN the auction for only 15 chips.
                #       We see their card on weak hands (when they can't defend).
                #   (b) When opp bids HIGH (strong hand, e.g. 251 chips):
                #       mean−1 = ~199 < 251 → they win, but pay ~199 into pot.
                #       Much better than old 15-chip bid (where they won for free).
                #   (c) As Welford learns their distribution, mean converges to
                #       their actual typical bid. We self-calibrate each hand.
                bid = max(1, int(self.opp_bid_mean) - 1)

            self.my_auction_bid          = int(min(bid, current_state.my_chips))
            self.my_chips_before_auction = current_state.my_chips   # snapshot for inference
            return ActionBid(self.my_auction_bid)

        # ── ML PREDICTION (unchanged from original) ───────────────────────────
        street   = current_state.street
        x_scaled = current_state.pot / 1000.0

        z   = (self.theta[street] * x_scaled) + self.b[street]
        y_hat = 1.0 / (1.0 + math.exp(-max(-50, min(50, z))))
        predicted_percentile = max(0.02, min(1.0, y_hat))

        self.current_hand_history[street] = {
            'x_scaled':     x_scaled,
            'y_hat':        y_hat,
            'cached_evals': None
        }

        # ── PRE-FLOP (unchanged from original) ───────────────────────────────
        if current_state.street == 'pre-flop':
            if current_state.cost_to_call > 0:
                if current_state.cost_to_call <= 20:
                    if self.preflop_score >= 9 and current_state.can_act(ActionRaise):
                        return ActionRaise(int(max(
                            current_state.raise_bounds[0],
                            min(current_state.raise_bounds[1],
                                current_state.raise_bounds[0] + 40))))
                    elif self.preflop_score >= 4:
                        return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                else:
                    if self.preflop_score >= 10:
                        if current_state.pot > 400 or self.preflop_score < 14:
                            return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                        if current_state.can_act(ActionRaise):
                            return ActionRaise(int(max(
                                current_state.raise_bounds[0],
                                min(current_state.raise_bounds[1],
                                    current_state.raise_bounds[0] + current_state.pot * 0.5))))
                        return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                    elif self.preflop_score >= 7 and current_state.cost_to_call < 200:
                        return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()
            else:
                if self.preflop_score >= 9 and current_state.can_act(ActionRaise):
                    return ActionRaise(int(current_state.raise_bounds[0]))
                return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()

        # ── POST-FLOP ML ENGINE ───────────────────────────────────────────────
        my_cards    = [eval7.Card(c) for c in current_state.my_hand]
        board_cards = tuple(eval7.Card(c) for c in current_state.board)

        hand_value = eval7.evaluate(my_cards + list(board_cards))
        hand_type  = eval7.handtype(hand_value)

        # Dead strings: our hand + board. DO NOT add revealed card here.
        # Revealed card must stay in deck so combos containing it can be built,
        # then we filter hand_strengths to those combos after building.
        dead_strings = set(current_state.my_hand + current_state.board)
        deck = [self.full_deck[cs] for cs in self.full_deck if cs not in dead_strings]

        hand_strengths = []
        for hand in itertools.combinations(deck, 2):
            hand_strengths.append((hand, eval7.evaluate(list(hand + board_cards))))
        hand_strengths.sort(key=lambda item: item[1], reverse=True)

        # If we won the auction and know one of their cards, restrict range
        revealed = current_state.opp_revealed_cards
        if revealed:
            rev_card_obj = self.full_deck.get(revealed[0])
            if rev_card_obj is not None:
                hand_strengths = [
                    (hand, s) for hand, s in hand_strengths if rev_card_obj in hand
                ]

        self.current_hand_history[street]['cached_evals'] = hand_strengths

        num_to_keep       = max(1, int(len(hand_strengths) * predicted_percentile))
        custom_eval7_range = [(h, 1.0) for h, _ in hand_strengths[:num_to_keep]]

        try:
            equity = eval7.py_hand_vs_range_monte_carlo(
                my_cards, custom_eval7_range, list(board_cards), 200)
        except Exception:
            equity = 0.0

        # Threat level from revealed card (original heuristic, preserved)
        threat_level = 0.0
        if revealed:
            board_ranks = [c[0] for c in current_state.board]
            board_suits = [c[1] for c in current_state.board]
            if revealed[0][0] in board_ranks:          threat_level += 0.25
            elif revealed[0][0] in ['A', 'K', 'Q']:   threat_level += 0.10
            if board_suits.count(revealed[0][1]) >= 2: threat_level += 0.10

        pot_odds = current_state.cost_to_call / max(1, current_state.pot + current_state.cost_to_call)

        # ── DECISION EXECUTION ────────────────────────────────────────────────
        safe_hand_types = ["Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"]

        can_raise_safely = True
        if current_state.pot > 800 and hand_type not in safe_hand_types:
            can_raise_safely = False

        if current_state.cost_to_call > 0:
            ctc = current_state.cost_to_call

            # ── FIX 3: GRADUATED OVERBET DEFENSE ─────────────────────────────
            # Original only folded at >1000 chips. These lower thresholds stop
            # slow chip bleeds from chasing with high card and weak pairs.
            if ctc > 400 and hand_type == "High Card" and equity < 0.45:
                return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()
            if ctc > 600 and hand_type == "Pair" and equity < 0.45:
                return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()
            if ctc > 1000 and hand_type in ["High Card", "Pair"]:
                return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()

            # ── FIX 2A: IRON-CLAD STRONG HAND PROTECTION ─────────────────────
            # ML can predict a tight opponent range → filtered equity drops low
            # → original code folds a flush or boat. That's a gifted pot.
            # Rule: with a made straight/flush/boat/quads → ALWAYS at least call.
            if hand_type in safe_hand_types:
                if (can_raise_safely and current_state.can_act(ActionRaise)
                        and equity > 0.65):
                    min_r, max_r = current_state.raise_bounds
                    target_bet   = max(min_r, min(max_r, min_r + int(current_state.pot * 0.75)))
                    return ActionRaise(int(target_bet))
                return ActionCall() if current_state.can_act(ActionCall) else ActionFold()

            # Three of a Kind: strong enough to defend at pot odds
            if hand_type == "Three of a Kind" and equity > 0.30:
                return ActionCall() if current_state.can_act(ActionCall) else ActionFold()

            required_equity = pot_odds + threat_level
            if equity > required_equity:
                # ── FIX 2B: ANTI-ESCALATION ───────────────────────────────────
                # Original raised with ANY hand at equity > required + 0.20.
                # In re-raise wars (two pair vs set, top pair vs two pair),
                # that escalates a 500-chip pot to 3000 chips and we often lose.
                # Only re-raise with genuinely nutted hands.
                can_reraise = False
                if can_raise_safely and current_state.can_act(ActionRaise):
                    if hand_type in safe_hand_types and equity > 0.65:
                        can_reraise = True
                    elif hand_type == "Three of a Kind" and equity > 0.80:
                        can_reraise = True

                if can_reraise:
                    min_r, max_r = current_state.raise_bounds
                    target_bet   = max(min_r, min(max_r, min_r + int(current_state.pot * 0.75)))
                    return ActionRaise(int(target_bet))
                return ActionCall() if current_state.can_act(ActionCall) else ActionFold()

            return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()

        else:
            # No bet to call: c-bet or check
            if equity > (0.55 + threat_level):
                if can_raise_safely and current_state.can_act(ActionRaise):
                    min_r, max_r = current_state.raise_bounds
                    target_bet   = max(min_r, min(max_r, min_r + int(current_state.pot * 0.50)))
                    return ActionRaise(int(target_bet))
            return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()


if __name__ == '__main__':
    run_bot(Player(), parse_args())