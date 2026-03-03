import random
import eval7
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

class Player(BaseBot):
    def __init__(self) -> None:
        self.hands_played = 0
        
        # 1. Pre-Compile C-Level Ranges for Zero Latency
        self.range_tight = eval7.HandRange("77+,A8s+,K9s+,QTs+,JTs+,ATo+,KJo+,QJo")
        self.range_medium = eval7.HandRange("44+,A2s+,K5s+,Q8s+,J8s+,T8s+,98s+,87s+,A7o+,K9o+,QTo+,JTo")
        # Explicit 100% PokerStove syntax to prevent eval7 from crashing on "XX"
        self.range_wide = eval7.HandRange("22+,A2s+,K2s+,Q2s+,J2s+,T2s+,92s+,82s+,72s+,62s+,52s+,42s+,32s+,A2o+,K2o+,Q2o+,J2o+,T2o+,92o+,82o+,72o+,62o+,52o+,42o+,32o+")
        
        # 2. Learning Mechanism (Opponent Baseline Aggression)
        self.avg_pot_size = 40.0 

    def _get_chen_score(self, cards):
        """Pre-flop heuristic: 0 latency, mathematically proven baseline."""
        ranks = {'A': 10, 'K': 8, 'Q': 7, 'J': 6, 'T': 5, '9': 4.5, '8': 4, '7': 3.5, '6': 3, '5': 2.5, '4': 2, '3': 1.5, '2': 1}
        rank1, suit1 = cards[0][0], cards[0][1]
        rank2, suit2 = cards[1][0], cards[1][1]
        
        score = max(ranks[rank1], ranks[rank2])
        if rank1 == rank2:
            score = max(5, ranks[rank1] * 2)
            
        if suit1 == suit2: score += 2
        
        gap = abs(ranks[rank1] - ranks[rank2])
        if gap == 1: score -= 1
        elif gap == 2: score -= 2
        elif gap == 3: score -= 4
        elif gap >= 4: score -= 5
        
        return score

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.hands_played += 1
        self.preflop_score = self._get_chen_score(current_state.my_hand)

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        # EMA Drag Fix: Only learn from hands that saw post-flop action.
        if current_state.pot > 40:
            self.avg_pot_size = (0.85 * self.avg_pot_size) + (0.15 * current_state.pot)

    def get_move(self, game_info: GameInfo, current_state: PokerState) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        # ---------------------------------------------------------
        # 1. AUCTION STREET (The Vickrey Tax)
        # ---------------------------------------------------------
        if current_state.street == 'auction':
            if self.preflop_score >= 8: bid_amt = 251 
            elif self.preflop_score >= 5: bid_amt = 248  
            else: bid_amt = 15   
            return ActionBid(int(min(bid_amt, current_state.my_chips)))

        # ---------------------------------------------------------
        # 2. PRE-FLOP LOGIC (Raise-Capped & Fallback Protected)
        # ---------------------------------------------------------
        if current_state.street == 'pre-flop':
            if current_state.cost_to_call > 0:
                if current_state.cost_to_call <= 20: 
                    # Small Blind / Min Raise completion
                    if self.preflop_score >= 9:
                        if current_state.can_act(ActionRaise):
                            min_r, max_r = current_state.raise_bounds
                            return ActionRaise(int(max(min_r, min(max_r, min_r + 40))))
                        return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                    elif self.preflop_score >= 4:
                        return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                else:
                    # Facing a heavy raise
                    if self.preflop_score >= 10:
                        # Pre-Flop Shove Cap: Never bloat massive pots without QQ, KK, AA
                        if current_state.pot > 400 or self.preflop_score < 14:
                            return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                            
                        if current_state.can_act(ActionRaise):
                            min_r, max_r = current_state.raise_bounds
                            return ActionRaise(int(max(min_r, min(max_r, min_r + current_state.pot * 0.5))))
                        return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                        
                    elif self.preflop_score >= 7 and current_state.cost_to_call < 200:
                        return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                        
                return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()
            else:
                if self.preflop_score >= 9 and current_state.can_act(ActionRaise):
                    return ActionRaise(int(current_state.raise_bounds[0]))
                return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()

        # ---------------------------------------------------------
        # 3. POST-FLOP LOGIC (Learned Range Monte Carlo)
        # ---------------------------------------------------------
        my_cards = [eval7.Card(c) for c in current_state.my_hand]
        board_cards = [eval7.Card(c) for c in current_state.board]
        
        # Determine opponent state dynamically using our EMA
        tight_threshold = self.avg_pot_size * 2.5
        medium_threshold = self.avg_pot_size * 1.2
        
        if current_state.pot > tight_threshold:
            assumed_range = self.range_tight
        elif current_state.pot > medium_threshold:
            assumed_range = self.range_medium
        else:
            assumed_range = self.range_wide
        
        # eval7 Blocker Crash Protection
        try:
            equity = eval7.py_hand_vs_range_monte_carlo(my_cards, assumed_range, board_cards, 200)
        except Exception:
            # If the range is mathematically empty because we hold the exact cards they need
            equity = 1.0

        # Threat Assessment from Auction Peek
        threat_level = 0.0
        if current_state.opp_revealed_cards:
            rev_card = current_state.opp_revealed_cards[0]
            board_ranks = [c[0] for c in current_state.board]
            board_suits = [c[1] for c in current_state.board]
            
            if rev_card[0] in board_ranks: threat_level += 0.25 
            elif rev_card[0] in ['A', 'K', 'Q']: threat_level += 0.10
            if board_suits.count(rev_card[1]) >= 2: threat_level += 0.10

        pot_size = current_state.pot
        pot_odds = current_state.cost_to_call / max(1, pot_size + current_state.cost_to_call)

        # Final EV Calculation and Move Execution
        if current_state.cost_to_call > 0:
            required_equity = pot_odds + threat_level
            
            if equity > required_equity:
                # Infinite Raise Loop Fix: Only re-raise if we are a crushing absolute favorite.
                if equity > max(0.65, required_equity + 0.20):
                    if current_state.can_act(ActionRaise):
                        min_r, max_r = current_state.raise_bounds
                        target_bet = max(min_r, min(max_r, min_r + int(pot_size * 0.75)))
                        return ActionRaise(int(target_bet))
                return ActionCall() if current_state.can_act(ActionCall) else ActionFold()
                
            return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()
            
        else:
            if equity > (0.55 + threat_level):
                if current_state.can_act(ActionRaise):
                    min_r, max_r = current_state.raise_bounds
                    target_bet = max(min_r, min(max_r, min_r + int(pot_size * 0.50)))
                    return ActionRaise(int(target_bet))
            return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()

if __name__ == '__main__':
    run_bot(Player(), parse_args())