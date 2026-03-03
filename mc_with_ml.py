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
        
        # 1. Scaled ML Parameters
        self.theta = {'pre-flop': 0.0, 'flop': 0.0, 'turn': 0.0, 'river': 0.0}
        self.b = {'pre-flop': 0.0, 'flop': 0.0, 'turn': 0.0, 'river': 0.0}
        self.learning_rate = 0.01 # Increased slightly since x is now scaled
        
        self.current_hand_history = {}

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
                    opp_card_str = opp_cards[0]
                    indices = [idx for idx, (hand, strength) in enumerate(hand_strengths) if opp_card_str in [c.__str__() for c in hand]]
                    y_true = (indices[len(indices)//2] / len(hand_strengths)) if indices else 0.5
                
            x_scaled = state_data['x_scaled']
            y_hat = state_data['y_hat']
            
            error = y_hat - y_true
            grad_sig = y_hat * (1.0 - y_hat)
            
            # Apply Gradient Descent
            self.theta[street] -= self.learning_rate * (error * grad_sig * x_scaled)
            self.b[street] -= self.learning_rate * (error * grad_sig)

    def get_move(self, game_info: GameInfo, current_state: PokerState) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        # 1. AUCTION STREET (Vickrey Tax)
        if current_state.street == 'auction':
            if self.preflop_score >= 8: bid_amt = 251 
            elif self.preflop_score >= 5: bid_amt = 248  
            else: bid_amt = 15   
            return ActionBid(int(min(bid_amt, current_state.my_chips)))

        street = current_state.street
        
        # Prevent MathOverflow: Scale Pot Size down for the ML Logic
        x_scaled = current_state.pot / 1000.0
        
        # 2. LOGISTIC REGRESSION PREDICTION
        z = (self.theta[street] * x_scaled) + self.b[street]
        y_hat = 1.0 / (1.0 + math.exp(-max(-50, min(50, z)))) # Clamped safety
        predicted_percentile = max(0.02, min(1.0, y_hat))

        self.current_hand_history[street] = {
            'x_scaled': x_scaled,
            'y_hat': y_hat,
            'cached_evals': None 
        }

        # 3. PRE-FLOP LOGIC (Strict Raise Caps)
        if current_state.street == 'pre-flop':
            if current_state.cost_to_call > 0:
                if current_state.cost_to_call <= 20: 
                    if self.preflop_score >= 9 and current_state.can_act(ActionRaise):
                        return ActionRaise(int(max(current_state.raise_bounds[0], min(current_state.raise_bounds[1], current_state.raise_bounds[0] + 40))))
                    elif self.preflop_score >= 4:
                        return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                else:
                    if self.preflop_score >= 10:
                        if current_state.pot > 400 or self.preflop_score < 14:
                            return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                        if current_state.can_act(ActionRaise):
                            return ActionRaise(int(max(current_state.raise_bounds[0], min(current_state.raise_bounds[1], current_state.raise_bounds[0] + current_state.pot * 0.5))))
                        return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                    elif self.preflop_score >= 7 and current_state.cost_to_call < 200:
                        return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()
            else:
                if self.preflop_score >= 9 and current_state.can_act(ActionRaise):
                    return ActionRaise(int(current_state.raise_bounds[0]))
                return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()

        # 4. POST-FLOP ML ENGINE
        my_cards = [eval7.Card(c) for c in current_state.my_hand]
        board_cards = tuple([eval7.Card(c) for c in current_state.board]) 
        
        hand_value = eval7.evaluate(my_cards + list(board_cards))
        hand_type = eval7.handtype(hand_value)

        # C-Engine Safety: Filter dead cards immediately to prevent Zero-Option Segfaults
        dead_strings = set(current_state.my_hand + current_state.board)
        deck = [self.full_deck[card_str] for card_str in self.full_deck if card_str not in dead_strings]
        
        possible_hands = itertools.combinations(deck, 2)
        hand_strengths = []
        for hand in possible_hands:
            hand_strengths.append((hand, eval7.evaluate(list(hand + board_cards))))
            
        hand_strengths.sort(key=lambda item: item[1], reverse=True)
        self.current_hand_history[street]['cached_evals'] = hand_strengths
        
        num_to_keep = max(1, int(len(hand_strengths) * predicted_percentile))
        
        # C-Engine Safety: Format exactly as (Tuple, Weight) to prevent option[0] unpack crash
        custom_eval7_range = [(hand_tuple, 1.0) for hand_tuple, strength in hand_strengths[:num_to_keep]]
        
        try:
            equity = eval7.py_hand_vs_range_monte_carlo(my_cards, custom_eval7_range, list(board_cards), 200)
        except Exception:
            equity = 0.0 # Exception Suicide Fix

        # Threat Assessment from Auction Peek
        threat_level = 0.0
        if current_state.opp_revealed_cards:
            rev_card = eval7.Card(current_state.opp_revealed_cards[0])
            their_known_cards = list(board_cards) + [rev_card]
            if len(their_known_cards) >= 5:
                if eval7.evaluate(their_known_cards) > hand_value:
                    equity = 0.0 
            
            board_ranks = [c[0] for c in current_state.board]
            board_suits = [c[1] for c in current_state.board]
            if current_state.opp_revealed_cards[0][0] in board_ranks: threat_level += 0.25 
            elif current_state.opp_revealed_cards[0][0] in ['A', 'K', 'Q']: threat_level += 0.10
            if board_suits.count(current_state.opp_revealed_cards[0][1]) >= 2: threat_level += 0.10

        pot_odds = current_state.cost_to_call / max(1, current_state.pot + current_state.cost_to_call)

        # 5. DECISION EXECUTION 
        can_raise_safely = True
        # String Based Cap: Fixes the bitmask bypass bug (Added "Flush" safety whitelist)
        if current_state.pot > 800 and hand_type not in ["Flush", "Full House", "Four of a Kind", "Straight Flush"]:
            can_raise_safely = False

        if current_state.cost_to_call > 0:
            # Massive Overbet Defense
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