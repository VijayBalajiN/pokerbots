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
        # ── ML Parameters (Intelligent Bayesian Priors) ─────────────────────────
        # Theta/b per street: model maps scaled pot → opponent hand-percentile
        self.theta = {'pre-flop': -1.0, 'flop': -1.0, 'turn': -1.0, 'river': -1.0}
        self.b     = {'pre-flop':  1.5, 'flop':  1.5, 'turn':  1.5, 'river':  1.5}
        self.learning_rate = 0.01

        self.current_hand_history = {}
        self.training_samples = 0  # counts only full 2-card reveals

        # ── Pot Scaling: linear up to 3200, soft log tail beyond ──────────────
        self._POT_SCALE = 400.0
        self._POT_CAP   = 8.0   # = 3200 / 400

        # ── Adaptive Auction: Opponent Bid Model (Welford) ───────────────────────
        # Tracks the distribution of the opponent's historical bids.
        # Seeded with a neutral prior (mean=50, std=50) before any data.
        self.opp_bid_n    = 1
        self.opp_bid_mean = 50.0
        self.opp_bid_M2   = 2500.0   # variance numerator; std = sqrt(M2 / n)

        # Per-hand auction state machine
        self.my_chips_before_auction = STARTING_STACK  # exact snapshot taken when we bid
        self.my_auction_bid          = 0               # what we bid this auction
        self.auction_resolved        = False           # True once we've processed the result
        # 'win'  = we actively tried to win (strong hand)
        # 'lose' = we deliberately bid below floor (weak/marginal hand)
        self.auction_intent          = 'lose'

        # ── Opponent Aggression Tracker ───────────────────────────────────────────
        # Counts post-flop bets/raises by opponent to inform check-raise decisions.
        self.opp_postflop_bets  = 0
        self.opp_postflop_hands = 0

        # ── Per-hand state ────────────────────────────────────────────────────────
        self.is_ip = False  # in position (acting last) this hand

        # ── Pre-allocated deck ────────────────────────────────────────────────────
        all_ranks = '23456789TJQKA'
        all_suits = 'cdhs'
        self.full_deck = {r + s: eval7.Card(r + s) for r in all_ranks for s in all_suits}
        self.preflop_percentiles = self._build_preflop_cache()

    # ─────────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────────

    def _get_chen_score(self, cards):
        """
        Modified Chen formula with corrections for known misevaluations:
        - Suited connectors / one-gappers: +1.5 / +0.5 (high playability,
          board coverage on low/mid textures that pure Chen ignores).
        - Offsuit dominated broadway (both ≥ T, one ≤ Q, not paired): −1.0
          (easily dominated by Ax, Kx combos; Chen overvalues these).
        """
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
        suited = suit1 == suit2
        if suited:
            score += 2
        gap = abs(ranks[rank1] - ranks[rank2])
        if   gap == 1: score -= 1
        elif gap == 2: score -= 2
        elif gap == 3: score -= 4
        elif gap >= 4: score -= 5

        # ── Playability corrections ──────────────────────────────────────────
        # Suited connectors (gap ≤ 1): high implied odds, strong board coverage
        # on low/mid flops that pure Chen misses entirely.
        if suited and gap <= 1 and rank1 != rank2:
            score += 1.5
        # Suited one-gappers (gap == 2): weaker but still playable
        elif suited and gap == 2:
            score += 0.5

        # Offsuit dominated broadway: KTo, QTo, KJo, QJo, JTo — these are
        # overvalued by Chen because of high card strength, but they play
        # terribly post-flop (dominated by Ax/Kx combos).
        if not suited and rank1 != rank2:
            r1, r2 = ranks[rank1], ranks[rank2]
            if min(r1, r2) >= 5 and max(r1, r2) <= 8:  # both T..Q range
                score -= 1.0

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

    # Discrete bet-sizing buckets: prevents inverse-engineering of equity from
    # a continuous sizing function.  Each bucket is selected probabilistically
    # so that no single bet size uniquely identifies our hand strength.
    _SIZE_SMALL  = 0.33   # probe / thin value
    _SIZE_MEDIUM = 0.75   # standard value bet
    _SIZE_LARGE  = 1.50   # polar (nuts or bluff)

    def _compute_bet_fraction(self, equity: float, predicted_percentile: float) -> float:
        """
        Returns a pot fraction using DISCRETE MIXED SIZING.

        Instead of a continuous slider (which leaks equity to tracking opponents),
        we select from 3 buckets {0.33, 0.75, 1.50} with probabilities that shift
        based on equity tier and opponent looseness.

        Against loose opponents: weight shifts toward larger buckets.
        Against tight opponents: weight shifts toward smaller buckets.
        """
        # ── Determine equity tier ─────────────────────────────────────────────
        # Opponent looseness shifts probability mass toward larger buckets.
        # loose_shift in [-0.15, +0.15]: positive = loose opp = bigger bets.
        loose_shift = (predicted_percentile - 0.5) * 0.30

        if equity >= 0.80:
            # Strong hand: mostly large, sometimes medium to disguise
            p_small  = max(0.0, 0.05 - loose_shift)
            p_medium = max(0.0, 0.35 - loose_shift)
            # p_large = remainder
        elif equity >= 0.60:
            # Medium hand: mostly medium, occasionally small or large
            p_small  = max(0.0, 0.25 - loose_shift)
            p_medium = 0.50
            # p_large = remainder
        else:
            # Thin value / marginal: mostly small, occasionally medium
            p_small  = max(0.0, 0.65 - loose_shift)
            p_medium = max(0.0, 0.30 + loose_shift * 0.5)
            # p_large = remainder

        # Ensure probabilities are valid
        p_small  = max(0.0, min(1.0, p_small))
        p_medium = max(0.0, min(1.0, p_medium))
        p_large  = max(0.0, 1.0 - p_small - p_medium)

        r = random.random()
        if r < p_small:
            return self._SIZE_SMALL
        elif r < p_small + p_medium:
            return self._SIZE_MEDIUM
        else:
            return self._SIZE_LARGE

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
            # We lost. How we estimate their bid depends on whether we *tried* to win.
            # If auction_intent == 'win': our bid was genuine → they beat it, so they
            #   bid somewhere above ours. Anchor on our bid + upward nudge.
            # If auction_intent == 'lose': our bid was deliberately near-zero and tells
            #   us nothing about their actual bid. Anchoring on it would cause a
            #   self-reinforcing downward spiral. Instead anchor on the current model
            #   mean, which is already our best estimate of their typical bid.
            opp_bid_std = self._welford_std(self.opp_bid_n, self.opp_bid_M2)
            if self.auction_intent == 'win':
                # Genuine contest: they outbid us, so they're somewhere above our bid
                soft_estimate = self.my_auction_bid + max(5.0, opp_bid_std * 0.5)
            else:
                # Intentional fold: use model mean as anchor (not our useless 0-bid)
                soft_estimate = self.opp_bid_mean + max(5.0, opp_bid_std * 0.25)
            soft_estimate = min(soft_estimate, 500.0)  # cap at reasonable maximum
            self._update_opp_bid_model(soft_estimate)

    # ─────────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────────

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.preflop_score   = self._get_chen_score(current_state.my_hand)
        self.my_chips_before_auction  = current_state.my_chips
        self.my_auction_bid           = 0
        self.auction_resolved         = False
        self.auction_intent           = 'lose'
        # In Heads-Up, the Big Blind acts FIRST post-flop (Out of Position).
        # SB (dealer) acts last post-flop = In Position.
        self.is_ip                    = not current_state.is_bb
        self.current_hand_history = {}
        # Per-hand aggression tracking (reset each hand)
        self.opp_was_aggressive_this_hand = False
        self.flop_seen_this_hand          = False

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

            # Decaying learning rate: stabilises after many training samples
            lr = self.learning_rate / (1.0 + self.training_samples * 0.01)
            self.theta[street] -= lr * (error * grad_sig * x_scaled)
            self.b[street]     -= lr * (error * grad_sig)

    # ─────────────────────────────────────────────────────────────────────────────
    # Decision Engine
    # ─────────────────────────────────────────────────────────────────────────────

    def get_move(self, game_info: GameInfo, current_state: PokerState) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:

        # ── Resolve auction result on first post-auction street ──────────────────
        if current_state.street in ('flop', 'turn', 'river'):
            self._resolve_auction(current_state)

        # Only count hands that actually reach the post-flop stage
        if current_state.street == 'flop' and not self.flop_seen_this_hand:
            self.opp_postflop_hands += 1
            self.flop_seen_this_hand = True

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
                target = opp_pred_max + 1 + random.randint(0, 5)
                bid_amt = int(min(target, current_state.my_chips))
                self.auction_intent = 'win'

            elif self.preflop_score >= 6:
                # ── MARGINAL HAND: decide based on information value ──────────────
                if opp_pred_max < 30 and random.random() < 0.40:
                    # Cheap contest: barely outbid their expected floor
                    target = opp_pred_min + 1 + random.randint(0, 3)
                    bid_amt = int(min(target, current_state.my_chips))
                    self.auction_intent = 'win'
                else:
                    # Bow out: bid below their floor so they win but pay their own high bid
                    bid_amt = max(0, int(opp_pred_min) - 1)
                    self.auction_intent = 'lose'

            else:
                # ── DON'T WANT TO WIN: bid below predicted floor ─────────────────
                bid_amt = max(0, int(opp_pred_min) - 1)
                self.auction_intent = 'lose'

            # Record for later chip-delta inference
            self.my_auction_bid          = int(min(bid_amt, current_state.my_chips))
            self.my_chips_before_auction = current_state.my_chips  # EXACT snapshot before paying
            return ActionBid(self.my_auction_bid)

        # =========================================================================
        # 2. LOGISTIC REGRESSION PREDICTION (linear + log tail scaling)
        # =========================================================================
        # x = pot/400              for pot ≤ 3200 (linear discrimination)
        # x = 8 + ln(1 + Δ/400)   for pot > 3200 (soft dampened tail)
        # Prevents cap arbitrage while keeping gradients bounded.
        street = current_state.street

        pot      = float(current_state.pot)
        x_raw    = pot / self._POT_SCALE
        if x_raw <= self._POT_CAP:
            x_scaled = x_raw
        else:
            # Soft logarithmic tail: preserves differentiation past the cap
            # pot=3200 → 8.0,  pot=5000 → ~9.70,  pot=10000 → ~11.25
            x_scaled = self._POT_CAP + math.log1p(x_raw - self._POT_CAP)

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

            # Position adjustment: tighten thresholds OOP, loosen IP
            pos_adj = -0.5 if self.is_ip else 0.5
            raise_threshold   += pos_adj
            call_threshold    += pos_adj
            premium_threshold += pos_adj

            if current_state.cost_to_call > 0:
                if current_state.cost_to_call <= 20:
                    if self.preflop_score >= raise_threshold and current_state.can_act(ActionRaise):
                        # Adaptive raise sizing: bigger vs loose, smaller vs tight
                        pf_frac  = self._compute_bet_fraction(0.70, predicted_percentile)
                        min_r, max_r = current_state.raise_bounds
                        target   = max(min_r, min(max_r, min_r + int(current_state.pot * pf_frac)))
                        return ActionRaise(int(target))
                    elif self.preflop_score >= call_threshold:
                        return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                else:
                    if self.preflop_score >= premium_threshold:
                        if current_state.pot > 400 or self.preflop_score < 14:
                            return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                        if current_state.can_act(ActionRaise):
                            pf_frac  = self._compute_bet_fraction(0.80, predicted_percentile)
                            min_r, max_r = current_state.raise_bounds
                            target   = max(min_r, min(max_r, min_r + int(current_state.pot * pf_frac)))
                            return ActionRaise(int(target))
                        return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()
            else:
                if self.preflop_score >= raise_threshold and current_state.can_act(ActionRaise):
                    pf_frac  = self._compute_bet_fraction(0.65, predicted_percentile)
                    min_r, max_r = current_state.raise_bounds
                    target   = max(min_r, min(max_r, min_r + int(current_state.pot * pf_frac)))
                    return ActionRaise(int(target))
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

        # Scale MC samples with range size: more samples for larger ranges
        num_mc_equity = min(500, max(100, len(filtered_range)))
        try:
            equity = eval7.py_hand_vs_range_monte_carlo(my_cards, filtered_range, list(board_cards), num_mc_equity)
        except Exception:
            equity = 0.0

        # Threat level: MC vs FULL revealed-card-constrained range (no ML filter)
        threat_level = 0.0
        if revealed and hand_strengths:
            full_range = [(hand_tuple, 1.0) for hand_tuple, _ in hand_strengths]
            num_mc_threat = min(300, max(50, len(full_range)))
            try:
                raw_equity   = eval7.py_hand_vs_range_monte_carlo(my_cards, full_range, list(board_cards), num_mc_threat)
                threat_level = max(0.0, 0.5 - raw_equity)
            except Exception:
                threat_level = 0.0

        pot_odds = current_state.cost_to_call / max(1, current_state.pot + current_state.cost_to_call)

        # =========================================================================
        # 5. DECISION EXECUTION
        # =========================================================================
        # Position adjustment for post-flop thresholds: IP can play slightly wider
        pos_edge = -0.03 if self.is_ip else 0.03

        can_raise_safely = True
        safe_hand_types = ["Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"]

        # Hard cap: don't raise into large pots without a strong made hand
        if current_state.pot > 800 and hand_type not in safe_hand_types:
            can_raise_safely = False

        # Exploitation override: after 10 quality training hands, relax cap
        if self.training_samples >= 10 and equity > 0.85:
            if hand_type in ["Three of a Kind", "Two Pair"]:
                can_raise_safely = True

        # ── Opponent aggression rate (for check-raise logic) ─────────────────────
        opp_aggression = (self.opp_postflop_bets / max(1, self.opp_postflop_hands))

        if current_state.cost_to_call > 0:
            # Only count one aggressive action per hand to keep the ratio cleanly between 0.0 and 1.0
            if not self.opp_was_aggressive_this_hand:
                self.opp_postflop_bets += 1
                self.opp_was_aggressive_this_hand = True

            # Overbet defence: now factors in equity + predicted range, not just hand type
            if current_state.cost_to_call > 1000:
                if equity < 0.35 and hand_type in ["High Card", "Pair"]:
                    return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()

            required_equity = pot_odds + threat_level + pos_edge
            if equity > required_equity:
                if can_raise_safely and current_state.can_act(ActionRaise) and equity > max(0.65, required_equity + 0.20):
                    min_r, max_r   = current_state.raise_bounds
                    bet_frac       = self._compute_bet_fraction(equity, predicted_percentile)
                    target_bet     = max(min_r, min(max_r, min_r + int(current_state.pot * bet_frac)))
                    return ActionRaise(int(target_bet))
                return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
            return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()
        else:
            # ── Slow-play / trap: check with the near-nuts 25% of the time ───────
            # This induces bluffs from aggressive opponents and disguises our range.
            if equity > 0.92 and hand_type in safe_hand_types and random.random() < 0.25:
                return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()

            # ── Check-raise trap: check with strong hands vs aggressive opponents ─
            # If opponent bets frequently (aggression > 40%), check strong hands
            # ~30% of the time to set up a check-raise on their next bet.
            if (opp_aggression > 0.40 and equity > 0.70
                    and hand_type not in ["High Card"]
                    and random.random() < 0.30):
                return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()

            # ML-Driven C-Betting: lower threshold against loose opponents
            bet_threshold = (0.40 if predicted_percentile > 0.60 else 0.55) + pos_edge

            if equity > (bet_threshold + threat_level):
                if can_raise_safely and current_state.can_act(ActionRaise):
                    min_r, max_r   = current_state.raise_bounds
                    bet_frac       = self._compute_bet_fraction(equity, predicted_percentile)
                    target_bet     = max(min_r, min(max_r, min_r + int(current_state.pot * bet_frac)))
                    return ActionRaise(int(target_bet))
            return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
