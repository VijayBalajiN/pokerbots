'''
All-In Bot: Always goes all-in on every street.
Used as a smoke test to verify the engine is working correctly.
'''
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot


class Player(BaseBot):
    '''
    A bot that always goes all-in (max raise) whenever possible,
    and bids the maximum in auctions.
    '''

    def __init__(self) -> None:
        self.hands_played = 0

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.hands_played += 1

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        pass

    def get_move(self, game_info: GameInfo, current_state: PokerState):
        # Auction: bid everything we have
        if current_state.street == 'auction':
            return ActionBid(current_state.my_chips)

        # If we can raise, go all-in (max raise)
        if current_state.can_act(ActionRaise):
            _min_raise, max_raise = current_state.raise_bounds
            return ActionRaise(max_raise)

        # If we can call, always call
        if current_state.can_act(ActionCall):
            return ActionCall()

        # Otherwise check
        if current_state.can_act(ActionCheck):
            return ActionCheck()

        # Fallback (should never reach here, but just in case)
        return ActionFold()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
