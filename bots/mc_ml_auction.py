import random
import itertools
import eval7
import math
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

class Player(BaseBot):
    def __init__(self) -> None:
        self.hands_played = 0
        
        # 1. Scaled ML Parameters (Intelligent Bayesian Priors)
        # Initializes expecting massive bets to mean tight ranges
        self.theta = {'pre-flop': -1.0, 'flop': -1.0, 'turn': -1.0, 'river': -1.0}
        self.b = {'pre-flop': 1.5, 'flop': 1.5, 'turn': 1.5, 'river': 1.5}
        self.learning_rate = 0.01
        
        self.current_hand_history = {}
        self.training_samples = 0 # Counts only full 2-card reveal events

        # 3. Learned Input Scaling (Welford Online Normalization, one set per street)
        # Priors seeded with reasonable pot-size estimates per street
        streets = ['pre-flop', 'flop', 'turn', 'river']
        self.scale_n    = {s: 1    for s in streets}
        self.scale_mean = {'pre-flop': 50.0, 'flop': 200.0, 'turn': 350.0, 'river': 500.0}
        self.scale_M2   = {'pre-flop': 2500.0, 'flop': 40000.0, 'turn': 122500.0, 'river': 250000.0}

        # 2. Latency Optimization: Pre-allocate deck & combinations
        all_ranks = '23456789TJQKA'
        all_suits = 'cdhs'
        self.full_deck = {r+s: eval7.Card(r+s) for r in all_ranks for s in all_suits}
        self.preflop_percentiles = self._build_preflop_cache()

    def _get_chen_score(self, cards):
        ranks = {'A': 10, 'K': 8, 'Q': 7, 'J': 6, 'T': 5, '9': 4.5, '8': 4, '7': 3.5, '6': 3, '5': 2.5, '4': 2, '3': 1.5, '2': 1}
        rank1, suit1 = cards[0][0], cards[0][1]
        rank2, suit2 = cards[1][0], cards[1][1]
        score = max(ranks[rank1], ranks[rank2])
        if rank1 == rank2: score = max(5, ranks[rank1] * 2)
        if suit1 == suit2: score += 2
        gap = abs(ranks[rank1] - ranks[rank2])
        if gap == 1: score -= 1
        elif gap == 2: score -= 2
        elif gap == 3: score -= 4
        elif gap >= 4: score -= 5
        return score

    def _build_preflop_cache(self):
        deck_strs = list(self.full_deck.keys())
        combos = list(itertools.combinations(deck_strs, 2))
        scored = []
        for c1, c2 in combos:
            scored.append((set([c1, c2]), self._get_chen_score([c1, c2])))
            
        scored.sort(key=lambda x: x[1], reverse=True)
        cache = {}
        total = len(scored)
        for idx, (combo_set, score) in enumerate(scored):
            cache[frozenset(combo_set)] = idx / total
        return cache

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.hands_played += 1
        self.preflop_score = self._get_chen_score(current_state.my_hand)
        self.current_hand_history = {} 

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        """The Latency-Safe Backpropagation Engine"""
        opp_cards = current_state.opp_revealed_cards
        if not opp_cards:
            return

        # Only full 2-card reveals count as quality supervised samples
        full_reveal = len(opp_cards) == 2
        if full_reveal:
            self.training_samples += 1
            
        for street, state_data in self.current_hand_history.items():
            if street == 'pre-flop':
                if len(opp_cards) == 2:
                    frozen = frozenset(opp_cards)
                    y_true = self.preflop_percentiles.get(frozen, 0.5)
                else:
                    y_true = 0.5 
            else:
                hand_strengths = state_data['cached_evals']
                if not hand_strengths:
                    continue
                    
                if len(opp_cards) == 2:
                    opp_set = set(opp_cards)
                    y_true = 0.5
                    for idx, (hand, strength) in enumerate(hand_strengths):
                        if set([c.__str__() for c in hand]) == opp_set:
                            y_true = idx / len(hand_strengths)
                            break
                else:
                    # Single revealed card: average the percentile over all hands containing it
                    # This gives an unbiased expectation rather than the biased median
                    opp_card_str = opp_cards[0]
                    indices = [idx for idx, (hand, strength) in enumerate(hand_strengths)
                               if opp_card_str in [c.__str__() for c in hand]]
                    y_true = (sum(indices) / len(indices) / len(hand_strengths)) if indices else 0.5
                
            x_scaled = state_data['x_scaled']
            y_hat = state_data['y_hat']
            
            error = y_hat - y_true
            grad_sig = y_hat * (1.0 - y_hat)
            
            # Apply Gradient Descent
            self.theta[street] -= self.learning_rate * (error * grad_sig * x_scaled)
            self.b[street] -= self.learning_rate * (error * grad_sig)

    def get_move(self, game_info: GameInfo, current_state: PokerState) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        # ---------------------------------------------------------
        # 1. ADAPTIVE AUCTION STRATEGY (Vickrey Exploit Defense)
        # ---------------------------------------------------------
        if current_state.street == 'auction':
            if self.preflop_score >= 10:
                # The Information Push: Randomize to prevent exact reading
                bid_amt = random.randint(150, 300)
            elif self.preflop_score >= 6:
                # The Trap Bid: Occasionally bid 0 to punish "Tax Farmers"
                if random.random() < 0.30:
                    bid_amt = 0
                else:
                    bid_amt = random.randint(15, 65)
            else:
                # Information Denial: Don't bleed chips for trash
                bid_amt = random.randint(0, 5)
                
            return ActionBid(int(min(bid_amt, current_state.my_chips)))

        # ---------------------------------------------------------
        # 2. LOGISTIC REGRESSION PREDICTION
        # ---------------------------------------------------------
        street = current_state.street
        
        # Learned Input Scaling: Welford online normalization (mean/std tracked per street)
        pot = float(current_state.pot)
        n = self.scale_n[street] + 1
        delta = pot - self.scale_mean[street]
        new_mean = self.scale_mean[street] + delta / n
        self.scale_M2[street] += delta * (pot - new_mean)
        self.scale_mean[street] = new_mean
        self.scale_n[street] = n
        std = math.sqrt(self.scale_M2[street] / n)
        x_scaled = (pot - new_mean) / max(std, 1.0)

        z = (self.theta[street] * x_scaled) + self.b[street]
        y_hat = 1.0 / (1.0 + math.exp(-max(-50, min(50, z)))) # Clamped safety
        predicted_percentile = max(0.02, min(1.0, y_hat))

        self.current_hand_history[street] = {
            'x_scaled': x_scaled,
            'y_hat': y_hat,
            'cached_evals': None 
        }

        # ---------------------------------------------------------
        # 3. PRE-FLOP LOGIC (Adaptive via Learned Opponent Percentile)
        # ---------------------------------------------------------
        if current_state.street == 'pre-flop':
            # predicted_percentile: low = opponent plays few hands (tight), high = plays many (loose).
            # tightness_adj > 0 means tight opponent → raise our requirements.
            # tightness_adj < 0 means loose opponent → lower requirements.
            # Multiplier 4.0 → max swing of ±2.0 across the [0.02, 1.0] percentile range.
            tightness_adj = (0.5 - predicted_percentile) * 4.0

            # Adaptive Chen score thresholds (clamped to sane ranges)
            raise_threshold   = max(6.0, min(12.0,  9.0 + tightness_adj))  # base 9
            call_threshold    = max(2.0, min(8.0,   5.0 + tightness_adj))  # base 5
            premium_threshold = max(8.0, min(14.0, 10.0 + tightness_adj))  # base 10

            if current_state.cost_to_call > 0:
                if current_state.cost_to_call <= 20:
                    # Small blind / limp: raise with good hands, call with playable ones
                    if self.preflop_score >= raise_threshold and current_state.can_act(ActionRaise):
                        return ActionRaise(int(max(current_state.raise_bounds[0], min(current_state.raise_bounds[1], current_state.raise_bounds[0] + 40))))
                    elif self.preflop_score >= call_threshold:
                        return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                else:
                    # Larger raise: require premium hand, scaled by how tight/loose they are
                    if self.preflop_score >= premium_threshold:
                        if current_state.pot > 400 or self.preflop_score < 14:
                            return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                        if current_state.can_act(ActionRaise):
                            return ActionRaise(int(max(current_state.raise_bounds[0], min(current_state.raise_bounds[1], current_state.raise_bounds[0] + current_state.pot * 0.5))))
                        return ActionCall() if current_state.can_act(ActionCall) else ActionFold()

                # Below threshold: fold rather than bleed chips
                return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()
            else:
                # No cost to call: open-raise with strong hands, check otherwise
                if self.preflop_score >= raise_threshold and current_state.can_act(ActionRaise):
                    return ActionRaise(int(current_state.raise_bounds[0]))
                return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()

        # ---------------------------------------------------------
        # 4. POST-FLOP ML ENGINE
        # ---------------------------------------------------------
        my_cards = [eval7.Card(c) for c in current_state.my_hand]
        board_cards = tuple([eval7.Card(c) for c in current_state.board]) 
        
        hand_value = eval7.evaluate(my_cards + list(board_cards))
        hand_type = eval7.handtype(hand_value)

        # C-Engine Safety: Filter dead cards immediately to prevent Zero-Option Segfaults
        revealed = current_state.opp_revealed_cards
        dead_strings = set(current_state.my_hand + current_state.board + (revealed if revealed else []))
        deck = [self.full_deck[card_str] for card_str in self.full_deck if card_str not in dead_strings]
        
        possible_hands = itertools.combinations(deck, 2)
        hand_strengths = []
        for hand in possible_hands:
            hand_strengths.append((hand, eval7.evaluate(list(hand + board_cards))))
            
        hand_strengths.sort(key=lambda item: item[1], reverse=True)

        # If we know one of their cards (auction peek), restrict range to only combos containing it
        if revealed:
            rev_card_obj = self.full_deck.get(revealed[0])
            if rev_card_obj is not None:
                hand_strengths = [(hand, strength) for hand, strength in hand_strengths
                                  if rev_card_obj in hand]
        self.current_hand_history[street]['cached_evals'] = hand_strengths
        
        num_to_keep = max(1, int(len(hand_strengths) * predicted_percentile))
        
        # C-Engine Safety: Format exactly as (Tuple, Weight) to prevent option[0] unpack crash
        custom_eval7_range = [(hand_tuple, 1.0) for hand_tuple, strength in hand_strengths[:num_to_keep]]
        
        try:
            equity = eval7.py_hand_vs_range_monte_carlo(my_cards, custom_eval7_range, list(board_cards), 200)
        except Exception:
            equity = 0.0 # Exception Suicide Fix

        # Threat Assessment: MC over the full revealed-card-constrained range (no ML filter)
        # This gives a fair, scale-invariant comparison vs our hand
        threat_level = 0.0
        if revealed and hand_strengths:
            full_reveal_range = [(hand_tuple, 1.0) for hand_tuple, _ in hand_strengths]
            try:
                raw_equity = eval7.py_hand_vs_range_monte_carlo(
                    my_cards, full_reveal_range, list(board_cards), 100)
                # threat_level rises as raw_equity falls below 0.5 (they're ahead of us)
                threat_level = max(0.0, 0.5 - raw_equity)
            except Exception:
                threat_level = 0.0

        pot_odds = current_state.cost_to_call / max(1, current_state.pot + current_state.cost_to_call)

        # ---------------------------------------------------------
        # 5. DECISION EXECUTION (With Dynamic Stop-Loss)
        # ---------------------------------------------------------
        can_raise_safely = True
        
        # The Original Hard Cap
        if current_state.pot > 800 and hand_type not in ["Flush", "Full House", "Four of a Kind", "Straight Flush"]:
            can_raise_safely = False

        # THE EXPLOITATION OVERRIDE (Dynamic Stop-Loss)
        # Only drop the shields if we have actually extracted ground-truth data from >= 10 hands
        if self.training_samples >= 10 and equity > 0.85:
            if hand_type in ["Straight", "Three of a Kind", "Two Pair"]:
                can_raise_safely = True

        if current_state.cost_to_call > 0:
            # Massive Overbet Defense (Always required to stop 5000 chip suicidal bluffs)
            if current_state.cost_to_call > 1000 and hand_type in ["High Card", "Pair"]:
                return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()

            required_equity = pot_odds + threat_level
            if equity > required_equity:
                if can_raise_safely and current_state.can_act(ActionRaise) and equity > max(0.65, required_equity + 0.20):
                    min_r, max_r = current_state.raise_bounds
                    target_bet = max(min_r, min(max_r, min_r + int(current_state.pot * 0.75)))
                    return ActionRaise(int(target_bet))
                return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                
            return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()
        else:
            if equity > (0.55 + threat_level):
                if can_raise_safely and current_state.can_act(ActionRaise):
                    min_r, max_r = current_state.raise_bounds
                    target_bet = max(min_r, min(max_r, min_r + int(current_state.pot * 0.50)))
                    return ActionRaise(int(target_bet))
            return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()

if __name__ == '__main__':
    run_bot(Player(), parse_args())