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

        # ── ML Parameters (Intelligent Bayesian Priors) ─────────────────────────
        # Theta/b per street: model maps scaled pot → opponent hand-percentile
        self.theta = {'pre-flop': -1.0, 'flop': -1.0, 'turn': -1.0, 'river': -1.0}
        self.b     = {'pre-flop':  1.5, 'flop':  1.5, 'turn':  1.5, 'river':  1.5}
        self.learning_rate = 0.01

        self.current_hand_history = {}
        self.training_samples = 0  # counts only full 2-card reveals

        # ── Welford Online Normalization (learned input scaling, per street) ─────
        streets = ['pre-flop', 'flop', 'turn', 'river']
        self.scale_n    = {s: 1    for s in streets}
        self.scale_mean = {'pre-flop':  50.0, 'flop': 200.0, 'turn': 350.0, 'river': 500.0}
        self.scale_M2   = {'pre-flop': 2500.0, 'flop': 40000.0, 'turn': 122500.0, 'river': 250000.0}

        # ── Adaptive Auction: Opponent Bid Model (Welford) ───────────────────────
        # Tracks the distribution of the opponent's historical bids.
        # Seeded with a neutral prior (mean=50, std=50) before any data.
        self.opp_bid_n    = 1
        self.opp_bid_mean = 50.0
        self.opp_bid_M2   = 2500.0   # variance numerator; std = sqrt(M2 / n)

        # Per-hand auction state machine
        self.my_initial_chips       = STARTING_STACK  # chips at hand start (post-blind)
        self.my_chips_before_auction = STARTING_STACK  # exact snapshot taken when we bid
        self.my_auction_bid         = 0               # what we bid this auction
        self.auction_resolved       = False           # True once we've processed the result

        # ── Pre-allocated deck ────────────────────────────────────────────────────
        all_ranks = '23456789TJQKA'
        all_suits = 'cdhs'
        self.full_deck = {r + s: eval7.Card(r + s) for r in all_ranks for s in all_suits}
        self.preflop_percentiles = self._build_preflop_cache()

    # ─────────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────────

    def _get_chen_score(self, cards):
        ranks = {
            'A': 10, 'K': 8, 'Q': 7, 'J': 6, 'T': 5,
            '9': 4.5, '8': 4, '7': 3.5, '6': 3, '5': 2.5,
            '4': 2,   '3': 1.5, '2': 1
        }
        rank1, suit1 = cards[0][0], cards[0][1]
        rank2, suit2 = cards[1][0], cards[1][1]
        score = max(ranks[rank1], ranks[rank2])
        if rank1 == rank2:
            score = max(5, ranks[rank1] * 2)
        if suit1 == suit2:
            score += 2
        gap = abs(ranks[rank1] - ranks[rank2])
        if   gap == 1: score -= 1
        elif gap == 2: score -= 2
        elif gap == 3: score -= 4
        elif gap >= 4: score -= 5
        return score

    def _build_preflop_cache(self):
        deck_strs = list(self.full_deck.keys())
        combos = list(itertools.combinations(deck_strs, 2))
        scored = [(set([c1, c2]), self._get_chen_score([c1, c2])) for c1, c2 in combos]
        scored.sort(key=lambda x: x[1], reverse=True)
        total = len(scored)
        cache = {}
        for idx, (combo_set, _) in enumerate(scored):
            cache[frozenset(combo_set)] = idx / total
        return cache

    def _welford_update(self, n, mean, M2, x):
        """One-step Welford update. Returns (new_n, new_mean, new_M2)."""
        n    += 1
        delta = x - mean
        mean += delta / n
        M2   += delta * (x - mean)
        return n, mean, M2

    def _welford_std(self, n, M2):
        """Population std estimate from Welford accumulators."""
        return math.sqrt(M2 / max(n, 1))

    def _update_opp_bid_model(self, bid_estimate):
        """Update the opponent bid distribution with a new (possibly soft) sample."""
        self.opp_bid_n, self.opp_bid_mean, self.opp_bid_M2 = self._welford_update(
            self.opp_bid_n, self.opp_bid_mean, self.opp_bid_M2, float(bid_estimate)
        )

    def _resolve_auction(self, current_state: PokerState):
        """
        Called once at the first post-auction street.
        Infers the opponent's bid from chip changes and updates the bid model.

        Auction payment rule (engine): winner pays the LOSER's bid.
          → If we WON:  my_chips dropped by opp_bid → opp_bid = initial_chips − current_chips
          → If we LOST: chips unchanged; we know opp_bid ≥ our_bid
        """
        if self.auction_resolved:
            return
        self.auction_resolved = True

        # Use the chip snapshot taken at auction time, not hand-start chips,
        # to exclude pre-flop betting from the delta calculation.
        chip_delta = self.my_chips_before_auction - current_state.my_chips
        we_won = bool(current_state.opp_revealed_cards)

        if we_won and chip_delta > 0:
            # Exact observation: opponent bid exactly chip_delta
            self._update_opp_bid_model(chip_delta)
        else:
            # We lost (or tied with 0-0): opponent bid ≥ our bid.
            # Soft update: assume their bid is our bid + half a std above the current mean
            # (conservative upward nudge; prevents the mean from being dragged low by losses).
            opp_bid_std = self._welford_std(self.opp_bid_n, self.opp_bid_M2)
            soft_estimate = self.my_auction_bid + max(5.0, opp_bid_std * 0.5)
            soft_estimate = min(soft_estimate, 500.0)  # cap at reasonable maximum
            self._update_opp_bid_model(soft_estimate)

    # ─────────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────────

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.hands_played += 1
        self.preflop_score   = self._get_chen_score(current_state.my_hand)
        self.my_initial_chips        = current_state.my_chips  # chips AFTER posting blind
        self.my_chips_before_auction  = current_state.my_chips
        self.my_auction_bid           = 0
        self.auction_resolved         = False
        self.current_hand_history = {}

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        """Backpropagation engine: updates θ/b for each street we acted on."""
        opp_cards = current_state.opp_revealed_cards
        if not opp_cards:
            return

        # Only full 2-card reveals count as clean supervised samples
        if len(opp_cards) == 2:
            self.training_samples += 1

        for street, state_data in self.current_hand_history.items():
            if street == 'pre-flop':
                if len(opp_cards) == 2:
                    y_true = self.preflop_percentiles.get(frozenset(opp_cards), 0.5)
                else:
                    y_true = 0.5  # weak signal; still update b lightly
            else:
                hand_strengths = state_data['cached_evals']
                if not hand_strengths:
                    continue

                if len(opp_cards) == 2:
                    opp_set = set(opp_cards)
                    y_true = 0.5
                    for idx, (hand, _) in enumerate(hand_strengths):
                        if {c.__str__() for c in hand} == opp_set:
                            y_true = idx / len(hand_strengths)
                            break
                else:
                    # Single card: unbiased expectation over all compatible hands
                    opp_card_str = opp_cards[0]
                    indices = [
                        idx for idx, (hand, _) in enumerate(hand_strengths)
                        if opp_card_str in [c.__str__() for c in hand]
                    ]
                    y_true = (sum(indices) / len(indices) / len(hand_strengths)) if indices else 0.5

            x_scaled = state_data['x_scaled']
            y_hat    = state_data['y_hat']
            error    = y_hat - y_true
            grad_sig = y_hat * (1.0 - y_hat)

            self.theta[street] -= self.learning_rate * (error * grad_sig * x_scaled)
            self.b[street]     -= self.learning_rate * (error * grad_sig)

    # ─────────────────────────────────────────────────────────────────────────────
    # Decision Engine
    # ─────────────────────────────────────────────────────────────────────────────

    def get_move(self, game_info: GameInfo, current_state: PokerState) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:

        # ── Resolve auction result on first post-auction street ──────────────────
        if current_state.street in ('flop', 'turn', 'river'):
            self._resolve_auction(current_state)

        # =========================================================================
        # 1. ADAPTIVE AUCTION STRATEGY
        # =========================================================================
        if current_state.street == 'auction':
            opp_bid_std = self._welford_std(self.opp_bid_n, self.opp_bid_M2)

            # Predicted opponent bid range
            opp_pred_min = max(0.0, self.opp_bid_mean - opp_bid_std)
            opp_pred_max = self.opp_bid_mean + opp_bid_std

            if self.preflop_score >= 10:
                # ── WANT TO WIN: barely outbid predicted max ─────────────────────
                # We have a strong hand; winning the auction gives us a card peek
                # that's worth paying for. We bid just above their likely ceiling
                # so we win but pay minimally (we pay THEIR bid, not ours).
                # Small random jitter prevents us from being read exactly.
                target = opp_pred_max + 1 + random.randint(0, 5)
                bid_amt = int(min(target, current_state.my_chips))

            elif self.preflop_score >= 6:
                # ── MARGINAL HAND: decide based on information value ──────────────
                # If opp is predicted to bid low (< 30), occasionally contest
                # cheaply (bid floor+1) — the peek is worth a small premium.
                # Otherwise stay below their floor to deny them cheap info.
                if opp_pred_max < 30 and random.random() < 0.40:
                    # Cheap contest: barely outbid their expected floor
                    target = opp_pred_min + 1 + random.randint(0, 3)
                    bid_amt = int(min(target, current_state.my_chips))
                else:
                    # Bow out: bid below their floor so they win but pay their own high bid
                    bid_amt = max(0, int(opp_pred_min) - 1)

            else:
                # ── DON'T WANT TO WIN: bid below predicted floor ─────────────────
                # Weak hand — information is not worth buying.
                # Bid below their floor: they win and pay their own high bid, we pay 0.
                # This is pure chip preservation.
                bid_amt = max(0, int(opp_pred_min) - 1)

            # Record for later chip-delta inference
            self.my_auction_bid          = int(min(bid_amt, current_state.my_chips))
            self.my_chips_before_auction = current_state.my_chips  # EXACT snapshot before paying
            return ActionBid(self.my_auction_bid)

        # =========================================================================
        # 2. LOGISTIC REGRESSION PREDICTION (Welford-normalised input)
        # =========================================================================
        street = current_state.street

        pot = float(current_state.pot)
        n   = self.scale_n[street] + 1
        delta    = pot - self.scale_mean[street]
        new_mean = self.scale_mean[street] + delta / n
        self.scale_M2[street]   += delta * (pot - new_mean)
        self.scale_mean[street]  = new_mean
        self.scale_n[street]     = n
        std      = math.sqrt(self.scale_M2[street] / n)
        x_scaled = (pot - new_mean) / max(std, 1.0)

        z     = (self.theta[street] * x_scaled) + self.b[street]
        y_hat = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))
        predicted_percentile = max(0.02, min(1.0, y_hat))

        self.current_hand_history[street] = {
            'x_scaled':    x_scaled,
            'y_hat':       y_hat,
            'cached_evals': None,
        }

        # =========================================================================
        # 3. PRE-FLOP LOGIC (Adaptive via Learned Opponent Percentile)
        # =========================================================================
        if current_state.street == 'pre-flop':
            # tightness_adj > 0  →  tight opp → raise our bar
            # tightness_adj < 0  →  loose opp → lower our bar
            tightness_adj     = (0.5 - predicted_percentile) * 4.0
            raise_threshold   = max(6.0, min(12.0,  9.0 + tightness_adj))
            call_threshold    = max(2.0, min(8.0,   5.0 + tightness_adj))
            premium_threshold = max(8.0, min(14.0, 10.0 + tightness_adj))

            if current_state.cost_to_call > 0:
                if current_state.cost_to_call <= 20:
                    if self.preflop_score >= raise_threshold and current_state.can_act(ActionRaise):
                        return ActionRaise(int(max(
                            current_state.raise_bounds[0],
                            min(current_state.raise_bounds[1], current_state.raise_bounds[0] + 40)
                        )))
                    elif self.preflop_score >= call_threshold:
                        return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                else:
                    if self.preflop_score >= premium_threshold:
                        if current_state.pot > 400 or self.preflop_score < 14:
                            return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                        if current_state.can_act(ActionRaise):
                            return ActionRaise(int(max(
                                current_state.raise_bounds[0],
                                min(current_state.raise_bounds[1],
                                    current_state.raise_bounds[0] + current_state.pot * 0.5)
                            )))
                        return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()
            else:
                if self.preflop_score >= raise_threshold and current_state.can_act(ActionRaise):
                    return ActionRaise(int(current_state.raise_bounds[0]))
                return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()

        # =========================================================================
        # 4. POST-FLOP ML ENGINE
        # =========================================================================
        my_cards   = [eval7.Card(c) for c in current_state.my_hand]
        board_cards = tuple(eval7.Card(c) for c in current_state.board)

        hand_value = eval7.evaluate(my_cards + list(board_cards))
        hand_type  = eval7.handtype(hand_value)

        # Build opponent range: exclude dead cards (our hand + board + revealed card)
        revealed    = current_state.opp_revealed_cards
        dead_strings = set(current_state.my_hand + current_state.board + (revealed if revealed else []))
        deck         = [self.full_deck[cs] for cs in self.full_deck if cs not in dead_strings]

        hand_strengths = []
        for hand in itertools.combinations(deck, 2):
            hand_strengths.append((hand, eval7.evaluate(list(hand + board_cards))))
        hand_strengths.sort(key=lambda item: item[1], reverse=True)

        # If we know one of their cards, restrict range to combos that include it
        if revealed:
            rev_card_obj = self.full_deck.get(revealed[0])
            if rev_card_obj is not None:
                hand_strengths = [
                    (hand, s) for hand, s in hand_strengths if rev_card_obj in hand
                ]

        self.current_hand_history[street]['cached_evals'] = hand_strengths

        # ML-filtered equity: play opponent as if they only hold top predicted_percentile
        num_to_keep     = max(1, int(len(hand_strengths) * predicted_percentile))
        filtered_range  = [(hand_tuple, 1.0) for hand_tuple, _ in hand_strengths[:num_to_keep]]

        try:
            equity = eval7.py_hand_vs_range_monte_carlo(my_cards, filtered_range, list(board_cards), 200)
        except Exception:
            equity = 0.0

        # Threat level: MC vs FULL revealed-card-constrained range (no ML filter)
        # Gives a scale-invariant measure of how far behind we are with that card known
        threat_level = 0.0
        if revealed and hand_strengths:
            full_range = [(hand_tuple, 1.0) for hand_tuple, _ in hand_strengths]
            try:
                raw_equity   = eval7.py_hand_vs_range_monte_carlo(my_cards, full_range, list(board_cards), 100)
                threat_level = max(0.0, 0.5 - raw_equity)
            except Exception:
                threat_level = 0.0

        pot_odds = current_state.cost_to_call / max(1, current_state.pot + current_state.cost_to_call)

        # =========================================================================
        # 5. DECISION EXECUTION
        # =========================================================================
        can_raise_safely = True

        # Hard cap: don't raise into large pots without a monster
        if current_state.pot > 800 and hand_type not in ["Flush", "Full House", "Four of a Kind", "Straight Flush"]:
            can_raise_safely = False

        # Exploitation override: after 10 quality training hands, relax cap on strong equities
        if self.training_samples >= 10 and equity > 0.85:
            if hand_type in ["Straight", "Three of a Kind", "Two Pair"]:
                can_raise_safely = True

        if current_state.cost_to_call > 0:
            # Overbet defence
            if current_state.cost_to_call > 1000 and hand_type in ["High Card", "Pair"]:
                return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()

            required_equity = pot_odds + threat_level
            if equity > required_equity:
                if can_raise_safely and current_state.can_act(ActionRaise) and equity > max(0.65, required_equity + 0.20):
                    min_r, max_r = current_state.raise_bounds
                    target_bet   = max(min_r, min(max_r, min_r + int(current_state.pot * 0.75)))
                    return ActionRaise(int(target_bet))
                return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
            return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()
        else:
            # ML-Driven C-Betting: if the model has learned the opponent plays a wide,
            # trashy range (high predicted_percentile), lower our betting threshold
            # so we bluff them off the pot even when we miss the flop.
            # Against a tight opponent (low percentile) keep the conservative 0.55 bar.
            bet_threshold = 0.40 if predicted_percentile > 0.60 else 0.55

            if equity > (bet_threshold + threat_level):
                if can_raise_safely and current_state.can_act(ActionRaise):
                    min_r, max_r = current_state.raise_bounds
                    target_bet   = max(min_r, min(max_r, min_r + int(current_state.pot * 0.50)))
                    return ActionRaise(int(target_bet))
            return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
