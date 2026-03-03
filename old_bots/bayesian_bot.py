import random
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

class Player(BaseBot):
    def __init__(self) -> None:
        self.hands_played = 0
        self.rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

    def _get_chen_score(self, cards):
        """Pre-flop heuristic: 0 latency."""
        ranks = {'A': 10, 'K': 8, 'Q': 7, 'J': 6, 'T': 5, '9': 4.5, '8': 4, '7': 3.5, '6': 3, '5': 2.5, '4': 2, '3': 1.5, '2': 1}
        rank1, suit1 = cards[0][0], cards[0][1]
        rank2, suit2 = cards[1][0], cards[1][1]
        
        score = max(ranks[rank1], ranks[rank2])
        if rank1 == rank2:
            score = ranks[rank1] * 2
            if score < 5: score = 5
        
        if suit1 == suit2: score += 2
        
        gap = abs(ranks[rank1] - ranks[rank2])
        if gap == 1: score -= 1
        elif gap == 2: score -= 2
        elif gap == 3: score -= 4
        elif gap >= 4: score -= 5
        
        return score

    def _evaluate_hand_and_draws(self, my_cards, board_cards):
        """Custom pure-Python evaluator. Mathematically scaled to match standard poker equity."""
        all_cards = my_cards + board_cards
        if not all_cards: return 0.20, False, False
        
        suits = [c[1] for c in all_cards]
        ranks = [self.rank_values[c[0]] for c in all_cards]
        
        counts = {r: ranks.count(r) for r in set(ranks)}
        freqs = sorted(counts.values(), reverse=True)
        
        suit_counts = {s: suits.count(s) for s in set(suits)}
        max_suit = max(suit_counts.values()) if suit_counts else 0
        is_flush = max_suit >= 5
        flush_draw = max_suit == 4
        
        unique_ranks = sorted(list(set(ranks)))
        if 14 in unique_ranks: unique_ranks.insert(0, 1) 
        
        is_straight = False
        straight_draw = False
        max_consecutive = 1
        consecutive = 1
        for i in range(1, len(unique_ranks)):
            if unique_ranks[i] == unique_ranks[i-1] + 1:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 1
                
        if max_consecutive >= 5: is_straight = True
        elif max_consecutive == 4: straight_draw = True
        
        # Corrected mathematical strength ladder
        strength = 0.35  # High Card baseline
        
        if is_flush and is_straight: 
            strength = 1.00  # Straight Flush
        elif freqs[0] == 4: 
            strength = 0.95  # Quads
        elif freqs[0] == 3 and len(freqs) > 1 and freqs[1] >= 2: 
            strength = 0.90  # Full House
        elif is_flush: 
            strength = 0.85  # Flush
        elif is_straight: 
            strength = 0.80  # Straight
        elif freqs[0] == 3: 
            strength = 0.75  # Trips
        elif freqs[0] == 2 and len(freqs) > 1 and freqs[1] >= 2: 
            strength = 0.70  # Two Pair
        elif freqs[0] == 2:
            hole_ranks = [self.rank_values[c[0]] for c in my_cards]
            board_ranks = [self.rank_values[c[0]] for c in board_cards] if board_cards else []
            
            if any(counts[r] >= 2 for r in hole_ranks):
                my_paired_rank = max([r for r in hole_ranks if counts[r] >= 2])
                if board_ranks and my_paired_rank >= max(board_ranks):
                    strength = 0.65  # Top Pair / Overpair
                else:
                    strength = 0.50  # Middle/Bottom Pair
            else:
                strength = 0.40  # Board is paired, we only have high card
                
        return strength, flush_draw, straight_draw

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.hands_played += 1
        self.preflop_score = self._get_chen_score(current_state.my_hand)

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        pass

    def get_move(self, game_info: GameInfo, current_state: PokerState) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        # ---------------------------------------------------------
        # 1. AUCTION STREET (Vickrey Troll Logic)
        # ---------------------------------------------------------
        if current_state.street == 'auction':
            if self.preflop_score >= 8:
                bid_amt = 251  # Overbid typical 249 to win premium hands
            elif self.preflop_score >= 5:
                bid_amt = 248  # The "Tax" bid for marginal hands
            else:
                bid_amt = 15   # Trap bid: bleeds 15 chips from opponent when we hold trash
                
            return ActionBid(min(bid_amt, current_state.my_chips))

        # ---------------------------------------------------------
        # 2. PRE-FLOP LOGIC
        # ---------------------------------------------------------
        if current_state.street == 'pre-flop':
            if current_state.cost_to_call > 0:
                if current_state.cost_to_call <= 20: 
                    # Smart Small Blind completion
                    if self.preflop_score >= 9 and current_state.can_act(ActionRaise):
                        min_r, max_r = current_state.raise_bounds
                        return ActionRaise(max(min_r, min(max_r, min_r + 40)))
                    elif self.preflop_score >= 4 and current_state.can_act(ActionCall):
                        return ActionCall()
                else:
                    # Facing a real raise
                    if self.preflop_score >= 10 and current_state.can_act(ActionRaise):
                        min_r, max_r = current_state.raise_bounds
                        return ActionRaise(max(min_r, min(max_r, min_r + int(current_state.pot * 0.5))))
                    elif self.preflop_score >= 7 and current_state.can_act(ActionCall):
                        return ActionCall()
                return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()
            else:
                if self.preflop_score >= 9 and current_state.can_act(ActionRaise):
                    return ActionRaise(current_state.raise_bounds[0])
                return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()

        # ---------------------------------------------------------
        # 3. POST-FLOP LOGIC (Dynamic Pot Odds)
        # ---------------------------------------------------------
        strength, flush_draw, straight_draw = self._evaluate_hand_and_draws(current_state.my_hand, current_state.board)
        
        if current_state.street in ['flop', 'turn']:
            if flush_draw: strength += 0.20
            if straight_draw: strength += 0.15

        # Refined Threat Level Logic
        threat_level = 0.0
        if current_state.opp_revealed_cards:
            rev_card = current_state.opp_revealed_cards[0]
            board_ranks = [c[0] for c in current_state.board]
            
            if rev_card[0] in board_ranks:
                threat_level += 0.25 # They definitively paired the board
            elif rev_card[0] in ['A', 'K']:
                threat_level += 0.15 # High Broadway threat
            elif rev_card[0] in ['Q', 'J']:
                threat_level += 0.10

        pot_size = current_state.pot
        pot_odds = current_state.cost_to_call / max(1, pot_size + current_state.cost_to_call)

        if current_state.cost_to_call > 0:
            # We call if our absolute hand strength beats the pot odds + threat margin
            required_equity = pot_odds + threat_level + 0.05
            
            if strength > required_equity:
                if current_state.can_act(ActionRaise) and strength > (required_equity + 0.25):
                    min_r, max_r = current_state.raise_bounds
                    target_bet = max(min_r, min(max_r, min_r + int(pot_size * 0.75)))
                    return ActionRaise(target_bet)
                return ActionCall()
                
            return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()

        else:
            # We have the initiative
            if strength > (0.60 + threat_level) and current_state.can_act(ActionRaise):
                min_r, max_r = current_state.raise_bounds
                target_bet = max(min_r, min(max_r, min_r + int(pot_size * 0.50)))
                return ActionRaise(target_bet)
            
            return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()

if __name__ == '__main__':
    run_bot(Player(), parse_args())