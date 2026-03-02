import random
import eval7
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

class Player(BaseBot):
    def __init__(self) -> None:
        self.hands_played = 0
        self.opp_auction_wins = 0
        self.rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

    def _get_chen_score(self, cards):
        """Instantly evaluates pre-flop hand strength (0 latency)."""
        ranks = {'A': 10, 'K': 8, 'Q': 7, 'J': 6, 'T': 5, '9': 4.5, '8': 4, '7': 3.5, '6': 3, '5': 2.5, '4': 2, '3': 1.5, '2': 1}
        rank1, suit1 = cards[0][0], cards[0][1]
        rank2, suit2 = cards[1][0], cards[1][1]
        
        score = max(ranks[rank1], ranks[rank2])
        if rank1 == rank2:
            score = ranks[rank1] * 2
            if score < 5: score = 5
        
        # Suited bonus
        if suit1 == suit2: score += 2
        
        # Gap penalty
        gap = abs(ranks[rank1] - ranks[rank2])
        if gap == 1: score -= 1
        elif gap == 2: score -= 2
        elif gap == 3: score -= 4
        elif gap >= 4: score -= 5
        
        return score

    def _check_draws(self, all_cards):
        """Heuristically detects open-ended straight draws and flush draws instantly."""
        suits = [c[1] for c in all_cards]
        # Get unique ranks and sort them
        ranks = sorted(list(set([self.rank_values[c[0]] for c in all_cards])))
        
        flush_draw = False
        straight_draw = False
        
        # Check Flush Draw (4 of the same suit)
        suit_counts = {s: suits.count(s) for s in set(suits)}
        if max(suit_counts.values()) == 4:
            flush_draw = True
            
        # Check Straight Draw (4 consecutive ranks)
        if 14 in ranks:
            ranks.insert(0, 1) # Ace can be low
            
        max_consecutive = 1
        current_consecutive = 1
        for i in range(1, len(ranks)):
            if ranks[i] == ranks[i-1] + 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
                
        if max_consecutive == 4:
            straight_draw = True
            
        return flush_draw, straight_draw

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.hands_played += 1
        self.preflop_score = self._get_chen_score(current_state.my_hand)

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        # Track if the opponent won the auction
        if current_state.opp_revealed_cards:
            self.opp_auction_wins += 1

    def get_move(self, game_info: GameInfo, current_state: PokerState) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        # ---------------------------------------------------------
        # 1. AUCTION STREET (Vickrey Exploit)
        # ---------------------------------------------------------
        if current_state.street == 'auction':
            # Info is most valuable when we have a marginal hand (score 6 to 9)
            if 6 <= self.preflop_score <= 9:
                bid_amt = int(min(15, current_state.my_chips * 0.01))
            else:
                # If we have a monster or total trash, the info won't change our play.
                bid_amt = 0 
                
            # If the opponent is obsessed with auctions (winning >60% of them), just bid 0 and let them bleed chips.
            if self.hands_played > 10 and (self.opp_auction_wins / self.hands_played) > 0.6:
                bid_amt = 0

            return ActionBid(min(bid_amt, current_state.my_chips))

        # ---------------------------------------------------------
        # 2. PRE-FLOP LOGIC
        # ---------------------------------------------------------
        if current_state.street == 'pre-flop':
            if current_state.cost_to_call > 0:
                if self.preflop_score >= 10 and current_state.can_act(ActionRaise):
                    return ActionRaise(current_state.raise_bounds[0])
                elif self.preflop_score >= 7 and current_state.can_act(ActionCall):
                    return ActionCall()
                return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()
            else:
                if self.preflop_score >= 9 and current_state.can_act(ActionRaise):
                    return ActionRaise(current_state.raise_bounds[0])
                return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()

        # ---------------------------------------------------------
        # 3. POST-FLOP LOGIC (Instant eval7 + Draw Heuristics + Threat Assessment)
        # ---------------------------------------------------------
        # Convert to eval7 objects for absolute strength eval
        e7_my_cards = [eval7.Card(c) for c in current_state.my_hand]
        e7_board_cards = [eval7.Card(c) for c in current_state.board]
        
        # Absolute hand strength (0 to 7462) normalized
        hand_value = eval7.evaluate(e7_my_cards + e7_board_cards)
        strength_ratio = hand_value / 7462.0 

        # 3a. Draw Blindness Fix (Boost score if we have a strong draw)
        if current_state.street in ['flop', 'turn']:
            flush_draw, straight_draw = self._check_draws(current_state.my_hand + current_state.board)
            if flush_draw:
                strength_ratio += 0.25
            if straight_draw:
                strength_ratio += 0.20

        # 3b. Revealed Card Threat Assessment
        threat_level = 0.0
        if current_state.opp_revealed_cards:
            rev_card = current_state.opp_revealed_cards[0]
            rev_rank, rev_suit = rev_card[0], rev_card[1]
            
            board_ranks = [c[0] for c in current_state.board]
            board_suits = [c[1] for c in current_state.board]
            
            # 1. Did they pair the board?
            if rev_rank in board_ranks:
                threat_level += 0.3
            # 2. Do they hold a high "Scare Card"?
            elif rev_rank in ['A', 'K', 'Q']:
                threat_level += 0.1
            # 3. Do they have a Flush Draw blocker?
            if board_suits.count(rev_suit) >= 2:
                threat_level += 0.15

        # 3c. Final Decision Logic
        pot_odds = current_state.cost_to_call / (current_state.pot + current_state.cost_to_call + 1)

        if current_state.cost_to_call > 0:
            # Facing a bet - adjust threshold by the threat level of their revealed card
            if strength_ratio > (0.6 + threat_level):
                if current_state.can_act(ActionRaise) and strength_ratio > 0.85:
                    return ActionRaise(current_state.raise_bounds[0])
                return ActionCall()
            
            # Good pot odds + decent hand
            if pot_odds < 0.2 and strength_ratio > (0.35 + threat_level):
                return ActionCall()
                
            return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()

        else:
            # We have the initiative
            if strength_ratio > (0.65 + threat_level) and current_state.can_act(ActionRaise):
                min_r, max_r = current_state.raise_bounds
                # Small value bet (10% into the max allowed raise)
                return ActionRaise(min_r + int((max_r - min_r) * 0.1)) 
            
            return ActionCheck()

if __name__ == '__main__':
    run_bot(Player(), parse_args())