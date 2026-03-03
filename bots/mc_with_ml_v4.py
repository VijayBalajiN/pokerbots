"""
mc_with_ml v4 — complete extraction engine upgrade.

═══════════════════════════════════════════════════════════════════════════
DIAGNOSIS FROM v3 RESULTS:
═══════════════════════════════════════════════════════════════════════════

mahoraga vs mc_with_ml_v3 (5 matches):
  Win rates:      49.7% vs 49.1%  — essentially equal
  Auction rates:  14.4% vs 14.2%  — essentially equal (Welford fixed this)
  Mahoraga avg win margin:   +12,443
  v3 avg win margin:          +2,309
  Ratio: 5.4× bigger wins for mahoraga

Both bots win the same percentage of hands and the same number of auctions.
The ENTIRE edge is chip extraction per won hand. Seven features remain
in mahoraga that v3 is missing — all ported here verbatim.

═══════════════════════════════════════════════════════════════════════════
7 FEATURES PORTED FROM MAHORAGA:
═══════════════════════════════════════════════════════════════════════════

1. MIXED BET SIZING (_compute_bet_fraction)
   v3 always bets 0.75x pot (raise) or 0.50x pot (c-bet).
   Mahoraga uses equity-tier buckets {0.33, 0.75, 1.50}× pot with random
   selection weighted by tier + opponent looseness.
   With equity=0.85: mainly 1.50× pot → 2× more chips extracted per strong hand.
   This single feature explains most of the 5.4× margin gap.

2. LOWER C-BET THRESHOLD VS LOOSE OPPONENTS
   v3: equity > 0.55 always.
   Mahoraga: equity > 0.40 when predicted_percentile > 0.60 (loose opp).
   More thin-value bets where they have positive EV.

3. ML PRIORS: theta=-1.0, b=1.5
   v3 starts at y_hat=0.50 (neutral). Mahoraga starts at y_hat=0.82 (loose).
   Immediate aggressive c-betting from hand 1 before the model learns.

4. DECEPTION: SLOW-PLAY + CHECK-RAISE
   25% slow-play (near-nuts vs aggressive opp): induces bluffs, builds pots.
   30% check-raise (strong hands vs aggressive opp): disguises range.

5. POSITION AWARENESS (is_ip)
   IP (SB/dealer, acts last post-flop): +0.03 threshold relaxation (wider play)
   OOP (BB, acts first post-flop): -0.03 (tighter play)

6. OPPONENT AGGRESSION TRACKER
   Tracks post-flop bet/raise frequency across hands.
   Drives the slow-play and check-raise decisions.

7. LOG-LINEAR POT SCALING
   v3: x_scaled = pot / 1000.0 (collapses large pots)
   Mahoraga: linear below 3200 (pot/400), log tail beyond.
   Better ML signal discrimination at all pot sizes.

═══════════════════════════════════════════════════════════════════════════
UNCHANGED FROM v3 (proven working):
═══════════════════════════════════════════════════════════════════════════
  - Welford adaptive auction (Fix 1 from v3)
  - Iron-clad strong hand protection (Fix 2a)
  - Anti-escalation / no re-raise with medium hands (Fix 2b)
  - Graduated overbet defense 400/600/1000 (Fix 3)
  - Decaying learning rate (Fix 4)
  - Unbiased y_true mean index (Fix 4)
  - Range-building logic (dead_strings = hand + board, no revealed)
  - Revealed-card range filter after building
  - Pre-flop logic (unchanged from original mc_with_ml)
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

        # FEATURE 3: Better priors — start assuming loose opponent.
        # theta=-1.0, b=1.5: at pot=0, z=1.5 → y_hat=0.82 (loose prior).
        # Immediately enables aggressive c-betting before the model trains.
        # v3 used 0.0/0.0 → y_hat=0.50 (neutral, too passive early game).
        self.theta = {'pre-flop': -1.0, 'flop': -1.0, 'turn': -1.0, 'river': -1.0}
        self.b     = {'pre-flop':  1.5, 'flop':  1.5, 'turn':  1.5, 'river':  1.5}
        self.learning_rate = 0.01
        self.training_samples = 0

        self.current_hand_history = {}

        # Welford adaptive auction (from v3, proven)
        self.opp_bid_n    = 1
        self.opp_bid_mean = 200.0
        self.opp_bid_M2   = 2500.0
        self.my_chips_before_auction = STARTING_STACK
        self.my_auction_bid          = 0
        self.auction_resolved        = False

        # FEATURE 6: Opponent aggression tracker
        self.opp_postflop_bets  = 0
        self.opp_postflop_hands = 0

        # FEATURE 5: Position tracking (set each hand)
        self.is_ip = False

        # Per-hand aggression flag (prevents double-counting)
        self.opp_was_aggressive_this_hand = False
        self.flop_seen_this_hand          = False

        # Pre-allocated deck
        all_ranks = '23456789TJQKA'
        all_suits = 'cdhs'
        self.full_deck = {r + s: eval7.Card(r + s) for r in all_ranks for s in all_suits}
        self.preflop_percentiles = self._build_preflop_cache()

        # FEATURE 7: Log-linear pot scaling constants
        self._POT_SCALE = 400.0
        self._POT_CAP   = 8.0   # = 3200 / 400

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

    # FEATURE 1: Mixed bet sizing (verbatim from mahoraga)
    _SIZE_SMALL  = 0.33
    _SIZE_MEDIUM = 0.75
    _SIZE_LARGE  = 1.50

    def _compute_bet_fraction(self, equity: float, predicted_percentile: float) -> float:
        """
        Discrete mixed sizing: {0.33, 0.75, 1.50}× pot, selected probabilistically.

        With equity ≥ 0.80: weight is mostly on 1.50× pot.
        → Against opponent who paid to see our card, this maximally extracts value.
        → v3's fixed 0.75× leaves 50% chips on table on every strong hand.

        loose_shift: positive = loose opponent = bigger bets get called more often.
        """
        loose_shift = (predicted_percentile - 0.5) * 0.30

        if equity >= 0.80:
            p_small  = max(0.0, 0.05 - loose_shift)
            p_medium = max(0.0, 0.35 - loose_shift)
        elif equity >= 0.60:
            p_small  = max(0.0, 0.25 - loose_shift)
            p_medium = 0.50
        else:
            p_small  = max(0.0, 0.65 - loose_shift)
            p_medium = max(0.0, 0.30 + loose_shift * 0.5)

        p_small  = max(0.0, min(1.0, p_small))
        p_medium = max(0.0, min(1.0, p_medium))
        p_large  = max(0.0, 1.0 - p_small - p_medium)

        r = random.random()
        if r < p_small:                return self._SIZE_SMALL
        elif r < p_small + p_medium:   return self._SIZE_MEDIUM
        else:                           return self._SIZE_LARGE

    def _resolve_auction(self, current_state: PokerState):
        if self.auction_resolved:
            return
        self.auction_resolved = True

        chip_delta = self.my_chips_before_auction - current_state.my_chips
        we_won = bool(current_state.opp_revealed_cards)

        if we_won and chip_delta > 0:
            self.opp_bid_n, self.opp_bid_mean, self.opp_bid_M2 = self._welford_update(
                self.opp_bid_n, self.opp_bid_mean, self.opp_bid_M2, float(chip_delta))
        else:
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
        # FEATURE 5: SB = dealer = acts last post-flop = In Position
        self.is_ip = not current_state.is_bb
        # Reset per-hand tracking
        self.opp_was_aggressive_this_hand = False
        self.flop_seen_this_hand          = False

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
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
                    opp_card_str = opp_cards[0]
                    indices = [idx for idx, (hand, _) in enumerate(hand_strengths)
                               if opp_card_str in [c.__str__() for c in hand]]
                    y_true = ((sum(indices) / len(indices)) / len(hand_strengths)
                              if indices else 0.5)

            x_scaled = state_data['x_scaled']
            y_hat    = state_data['y_hat']
            error    = y_hat - y_true
            grad_sig = y_hat * (1.0 - y_hat)

            lr = self.learning_rate / (1.0 + self.training_samples * 0.01)
            self.theta[street] -= lr * (error * grad_sig * x_scaled)
            self.b[street]     -= lr * (error * grad_sig)

    # ── main entry ────────────────────────────────────────────────────────────

    def get_move(self, game_info: GameInfo, current_state: PokerState):

        if current_state.street in ('flop', 'turn', 'river'):
            self._resolve_auction(current_state)

        # FEATURE 6: Count hands reaching post-flop for aggression ratio
        if current_state.street == 'flop' and not self.flop_seen_this_hand:
            self.opp_postflop_hands += 1
            self.flop_seen_this_hand = True

        # ── ADAPTIVE WELFORD AUCTION ──────────────────────────────────────────
        if current_state.street == 'auction':
            opp_std      = self._welford_std(self.opp_bid_n, self.opp_bid_M2)
            opp_pred_max = self.opp_bid_mean + opp_std

            if self.preflop_score >= 10:
                bid = int(opp_pred_max + 1 + random.randint(0, 5))
            else:
                bid = max(1, int(self.opp_bid_mean) - 1)

            self.my_auction_bid          = int(min(bid, current_state.my_chips))
            self.my_chips_before_auction = current_state.my_chips
            return ActionBid(self.my_auction_bid)

        # ── ML PREDICTION ────────────────────────────────────────────────────
        street = current_state.street

        # FEATURE 7: Log-linear pot scaling
        pot   = float(current_state.pot)
        x_raw = pot / self._POT_SCALE
        if x_raw <= self._POT_CAP:
            x_scaled = x_raw
        else:
            x_scaled = self._POT_CAP + math.log1p(x_raw - self._POT_CAP)

        z     = (self.theta[street] * x_scaled) + self.b[street]
        y_hat = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))
        predicted_percentile = max(0.02, min(1.0, y_hat))

        self.current_hand_history[street] = {
            'x_scaled':     x_scaled,
            'y_hat':        y_hat,
            'cached_evals': None
        }

        # ── PRE-FLOP (unchanged from original mc_with_ml) ────────────────────
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

        # Dead strings: our hand + board only. Never add revealed card here
        # (revealed card must be in deck so combos containing it can be built).
        dead_strings = set(current_state.my_hand + current_state.board)
        deck = [self.full_deck[cs] for cs in self.full_deck if cs not in dead_strings]

        hand_strengths = []
        for hand in itertools.combinations(deck, 2):
            hand_strengths.append((hand, eval7.evaluate(list(hand + board_cards))))
        hand_strengths.sort(key=lambda item: item[1], reverse=True)

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

        # Threat level from revealed card (original heuristic)
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
        if self.training_samples >= 10 and equity > 0.85:
            if hand_type == "Three of a Kind":
                can_raise_safely = True

        # FEATURE 5: Position edge on equity thresholds
        pos_edge = -0.03 if self.is_ip else 0.03

        # FEATURE 6: Opponent aggression rate
        opp_aggression = self.opp_postflop_bets / max(1, self.opp_postflop_hands)

        if current_state.cost_to_call > 0:
            ctc = current_state.cost_to_call

            # FEATURE 6: Track one aggressive action per hand
            if not self.opp_was_aggressive_this_hand:
                self.opp_postflop_bets += 1
                self.opp_was_aggressive_this_hand = True

            # Graduated overbet defense (from v3)
            if ctc > 400 and hand_type == "High Card" and equity < 0.45:
                return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()
            if ctc > 600 and hand_type == "Pair" and equity < 0.45:
                return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()
            if ctc > 1000 and hand_type in ["High Card", "Pair"]:
                return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()

            # Iron-clad strong hand protection (from v3)
            if hand_type in safe_hand_types:
                if (can_raise_safely and current_state.can_act(ActionRaise)
                        and equity > 0.65):
                    min_r, max_r = current_state.raise_bounds
                    # FEATURE 1: Mixed sizing — can go 1.50x on the nuts
                    bet_frac   = self._compute_bet_fraction(equity, predicted_percentile)
                    target_bet = max(min_r, min(max_r, min_r + int(current_state.pot * bet_frac)))
                    return ActionRaise(int(target_bet))
                return ActionCall() if current_state.can_act(ActionCall) else ActionFold()

            if hand_type == "Three of a Kind" and equity > 0.30:
                return ActionCall() if current_state.can_act(ActionCall) else ActionFold()

            required_equity = pot_odds + threat_level + pos_edge
            if equity > required_equity:
                # Anti-escalation: only re-raise with nutted hands
                can_reraise = False
                if can_raise_safely and current_state.can_act(ActionRaise):
                    if hand_type in safe_hand_types and equity > 0.65:
                        can_reraise = True
                    elif hand_type == "Three of a Kind" and equity > 0.80:
                        can_reraise = True

                if can_reraise:
                    min_r, max_r = current_state.raise_bounds
                    # FEATURE 1: Mixed sizing on re-raises too
                    bet_frac   = self._compute_bet_fraction(equity, predicted_percentile)
                    target_bet = max(min_r, min(max_r, min_r + int(current_state.pot * bet_frac)))
                    return ActionRaise(int(target_bet))
                return ActionCall() if current_state.can_act(ActionCall) else ActionFold()

            return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()

        else:
            # ── FEATURE 4A: SLOW-PLAY TRAP ───────────────────────────────────
            # Check near-nuts ~25% of time ONLY against aggressive opponents.
            # Passive opponents: betting > checking (they won't bet into us).
            if (equity > 0.92 and hand_type in safe_hand_types
                    and opp_aggression > 0.40 and random.random() < 0.25):
                return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()

            # ── FEATURE 4B: CHECK-RAISE TRAP ─────────────────────────────────
            # Check strong hands ~30% of time vs aggressive opponents.
            # They bet, we raise → bigger pot on our terms.
            if (opp_aggression > 0.40 and equity > 0.70
                    and hand_type not in ["High Card"]
                    and random.random() < 0.30):
                return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()

            # ── FEATURE 2: LOWER C-BET THRESHOLD VS LOOSE OPPONENTS ──────────
            # Loose opponents (predicted_percentile > 0.60) call too wide.
            # Bet 0.40 threshold extracts thin value they'd never see at 0.55.
            bet_threshold = (0.40 if predicted_percentile > 0.60 else 0.55) + pos_edge

            if equity > (bet_threshold + threat_level):
                if can_raise_safely and current_state.can_act(ActionRaise):
                    min_r, max_r = current_state.raise_bounds
                    # FEATURE 1: Mixed sizing on c-bets too
                    bet_frac   = self._compute_bet_fraction(equity, predicted_percentile)
                    target_bet = max(min_r, min(max_r, min_r + int(current_state.pot * bet_frac)))
                    return ActionRaise(int(target_bet))
            return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()


if __name__ == '__main__':
    run_bot(Player(), parse_args())