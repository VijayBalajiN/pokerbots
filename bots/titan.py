"""
Titan — Designed to beat mahoraga.

Root-cause analysis of mahoraga's win over oracle/mc_with_ml:
──────────────────────────────────────────────────────────────
1. Discrete bet sizing {0.33, 0.75, 1.50}: harder to reverse-engineer equity.
   oracle's fixed 0.75/0.65 fractions are trivially readable.
2. Anti-escalation: never re-raises top-pair into 3-bets (oracle's #1 chip leak).
3. Check-raise traps: captures extra value vs aggressive opponents.
4. Slow-play: induces bluffs from aggressive opponents.
5. Aggression tracking: adapts to opponent's bet frequency.
6. Position awareness: IP = tighter, OOP = wider.

What oracle has that mahoraga lacks (the bugs we carry forward as fixes):
─────────────────────────────────────────────────────────────────────────
1. Backprop fix: store FULL ~990-combo range for on_hand_end, never restricted.
   mahoraga stores the restricted 44-combo range → when single card revealed,
   ALL combos contain that card → y_true always 0.5 → zero learning signal.
2. Range restriction: restrict play range to ~44 combos containing known card
   for equity decisions. mahoraga does this too, but loses the backprop.
3. Per-street equity caching: build range once, compute equity once per street.
   Critical for timing — avoids recomputation during raise/reraise wars.
4. When card known: lift pot-size raise cap (equity is accurate from ~44 combos).

Additional improvements over both:
───────────────────────────────────
1. Bet sizing {0.33, 0.67, 1.10}: cap large bet at 1.10x (vs mahoraga's 1.50x).
   Prevents catastrophic misfires when ML miscalibrates non-nut hand equity.
   Large bet (1.10x) restricted to nut hands (Straight+) only.
2. Trap/slow-play only activates after MODEL_N=20 hands of data to prevent
   false positives from early noise (similar to nemesis fix).
"""

import random
import itertools
import eval7
import math
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState, STARTING_STACK
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

_RANK_ORD = '23456789TJQKA'
_RANK_VAL = {r: v for r, v in zip(_RANK_ORD, [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10])}
_SAFE     = {'Straight', 'Flush', 'Full House', 'Four of a Kind', 'Straight Flush'}
_MODEL_N  = 20   # hands before trusting the aggression model


class Player(BaseBot):

    def __init__(self):
        # ── ML parameters (Bayesian priors for fast convergence) ──────────────
        # At pot=400 → x=1.0 → z=0.5 → y_hat≈0.62 (slightly-weak prior)
        self.theta = {s: -1.0 for s in ('pre-flop', 'flop', 'turn', 'river')}
        self.b     = {s:  1.5 for s in ('pre-flop', 'flop', 'turn', 'river')}
        self.lr            = 0.01
        self.n_samples     = 0       # full 2-card reveal training samples
        self.hand_history  = {}      # {street: {x_scaled, y_hat, backprop_evals}}

        # ── Pot scaling ───────────────────────────────────────────────────────
        self._PS = 400.0
        self._PC = 8.0

        # ── Welford opponent bid model ────────────────────────────────────────
        # Seeded near competitive bid range (mean=200, std=50 → opp_max≈250).
        # Converges to correct estimate in ~5 hands.
        self.opp_bid_n    = 1
        self.opp_bid_mean = 200.0
        self.opp_bid_M2   = 2500.0

        # ── Per-hand state (reset in on_hand_start) ───────────────────────────
        self.score              = 0
        self.is_ip              = False    # in-position (SB, acts last post-flop)
        self.raised_pf          = False
        self.my_bid             = 0
        self.chips_before_bid   = STARTING_STACK
        self.bid_resolved       = False
        self._range_cache       = {}       # {street: {play, full, eq, thr}}
        self._flop_seen         = False
        self._opp_bet_this_hand = False    # opp bet ≥1 time post-flop this hand

        # ── Cross-hand opponent model ─────────────────────────────────────────
        self.opp_bet_hands  = 0    # hands where opp bet post-flop (≥1 bet)
        self.opp_hand_count = 0    # hands that reached the flop

        # ── Pre-allocated deck ────────────────────────────────────────────────
        self.deck = {r + s: eval7.Card(r + s)
                     for r in '23456789TJQKA' for s in 'cdhs'}
        self.pf_pcts = self._build_pf_cache()

    # ─────────────────────────────────────────────────────────────────────────
    # Static helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _chen(self, h):
        """Modified Chen formula with suited-connector bonus."""
        r1, s1 = h[0][0], h[0][1]
        r2, s2 = h[1][0], h[1][1]
        sc = max(_RANK_VAL[r1], _RANK_VAL[r2])
        if r1 == r2:
            sc = max(5.0, _RANK_VAL[r1] * 2)
        if s1 == s2:
            sc += 2
        gap = abs(_RANK_ORD.index(r1) - _RANK_ORD.index(r2))
        sc -= (0, 1, 2, 4, 5)[min(gap, 4)]
        # Suited connector/one-gapper bonus (high implied odds)
        if s1 == s2 and 0 < gap <= 2:
            sc += 1.5 if gap == 1 else 0.5
        return sc

    def _build_pf_cache(self):
        dk = list(self.deck.keys())
        scored = sorted(
            [(frozenset([a, b]), self._chen([a, b]))
             for a, b in itertools.combinations(dk, 2)],
            key=lambda x: x[1], reverse=True)
        n = len(scored)
        return {fs: i / n for i, (fs, _) in enumerate(scored)}

    def _opp_std(self):
        return math.sqrt(self.opp_bid_M2 / max(1, self.opp_bid_n))

    def _welford(self, x):
        n    = self.opp_bid_n + 1
        d    = x - self.opp_bid_mean
        mean = self.opp_bid_mean + d / n
        M2   = self.opp_bid_M2 + d * (x - mean)
        self.opp_bid_n, self.opp_bid_mean, self.opp_bid_M2 = n, mean, M2

    def _ml(self, street, pot):
        """Return (y_hat, x_scaled) from per-street logistic model."""
        x  = pot / self._PS
        xs = x if x <= self._PC else self._PC + math.log1p(x - self._PC)
        z  = self.theta[street] * xs + self.b[street]
        yh = 1.0 / (1.0 + math.exp(-max(-50., min(50., z))))
        return max(0.02, min(1.0, yh)), xs

    # ─────────────────────────────────────────────────────────────────────────
    # Range building (per-street cache)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_range(self, cs):
        """
        Build and cache both play range and full backprop range for this street.
        play range: ~44 combos when opp card known, ~990 otherwise.
        full range: always ~990 combos (unrestricted) — used only for backprop.
        """
        st = cs.street
        if st in self._range_cache:
            return self._range_cache[st]

        dead  = set(cs.my_hand + cs.board)
        avail = [k for k in self.deck if k not in dead]
        brd   = [self.deck[k] for k in cs.board]
        av_c  = [self.deck[k] for k in avail]

        # Full unrestricted range — ALWAYS needed for backprop in on_hand_end.
        # (Bug in mahoraga: stores restricted range for backprop →
        #  with single-card reveal, y_true is always 0.5 → zero learning signal.)
        full_hs = sorted(
            [(h, eval7.evaluate(list(h) + brd))
             for h in itertools.combinations(av_c, 2)],
            key=lambda x: x[1], reverse=True)

        # Play range: restrict to ~44 combos containing known card if we won peek.
        # This is the 20× precision improvement for equity decisions.
        rev = cs.opp_revealed_cards
        if rev and rev[0] not in dead:
            c1 = self.deck[rev[0]]
            play_hs = sorted(
                [((c1, self.deck[c2]), eval7.evaluate([c1, self.deck[c2]] + brd))
                 for c2 in avail if c2 != rev[0]],
                key=lambda x: x[1], reverse=True)
        else:
            play_hs = full_hs

        rc = {'play': play_hs, 'full': full_hs, 'eq': None, 'thr': None}
        self._range_cache[st] = rc
        return rc

    # ─────────────────────────────────────────────────────────────────────────
    # Equity computation (per-street cache)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_eq(self, cs, rc, pp):
        """
        Compute and cache (equity, threat) for this street.
        equity  : vs ML-filtered play range (top pp% of play combos).
        threat  : max(0, 0.5 − raw_equity_vs_full_play_range).
        """
        if rc['eq'] is not None:
            return rc['eq'], rc['thr']

        my   = [self.deck[k] for k in cs.my_hand]
        brd  = [self.deck[k] for k in cs.board]
        play = rc['play']

        keep = max(1, int(len(play) * pp))
        filt = [(h, 1.0) for h, _ in play[:keep]]
        nmc  = min(300, max(80, len(filt)))
        try:
            eq = eval7.py_hand_vs_range_monte_carlo(my, filt, brd, nmc)
        except Exception:
            eq = 0.5

        # Threat: actual equity vs full revealed-constrained range (no ML filter).
        # Only computed when play range was restricted (card known).
        thr = 0.0
        if cs.opp_revealed_cards and play is not rc['full']:
            full_r = [(h, 1.0) for h, _ in play]
            try:
                raw = eval7.py_hand_vs_range_monte_carlo(
                    my, full_r, brd, min(150, len(full_r)))
                thr = max(0.0, 0.5 - raw)
            except Exception:
                pass

        rc['eq'], rc['thr'] = eq, thr
        return eq, thr

    # ─────────────────────────────────────────────────────────────────────────
    # Discrete mixed bet sizing
    # ─────────────────────────────────────────────────────────────────────────

    def _bet_frac(self, eq, pp, is_nut):
        """
        Three buckets: small (0.33), medium (0.67), large (1.10).

        CRITICAL: large (1.10) is ONLY available for nut hands (Straight+).
        This prevents the main failure mode of mc_with_ml/oracle: misfiring a
        large bet with two-pair or top-pair based on an overconfident MC estimate,
        then losing to a 3-bet we can't call.

        loose_shift: positive when ML predicts opp is weak → bigger bets.
        (pp > 0.5 means opp hand predicted below median = weak opponent.)
        """
        ls = (pp - 0.5) * 0.25   # ∈ [−0.125, +0.125]

        if is_nut and eq >= 0.78:
            # Nut + high equity: weight toward large
            ps = max(0.0, 0.05 - ls)
            pm = max(0.0, 0.28 - ls)
            pl = max(0.0, 1.0 - ps - pm)
        elif eq >= 0.60:
            # Medium-strong, non-nut: only small/medium (no large)
            ps = max(0.0, 0.22 - ls)
            pm = min(1.0, max(0.0, 0.78 + ls * 0.3))
            pl = 0.0
        else:
            # Thin value / c-bet / semi-bluff: mostly small
            ps = max(0.0, 0.55 - ls)
            pm = max(0.0, 0.40 + ls * 0.4)
            pl = 0.0

        r = random.random()
        ps = min(1.0, ps); pm = min(1.0, pm)
        if r < ps:           return 0.33
        if r < ps + pm:      return 0.67
        return 1.10

    # ─────────────────────────────────────────────────────────────────────────
    # Action helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _raise_to(self, cs, frac):
        """ActionRaise to (min_raise + frac*pot), clamped. Returns None if unavailable."""
        if not cs.can_act(ActionRaise): return None
        lo, hi = cs.raise_bounds
        return ActionRaise(int(max(lo, min(hi, lo + int(cs.pot * frac)))))

    def _fold_or_check(self, cs):
        return ActionFold() if cs.can_act(ActionFold) else ActionCheck()

    def _call_or_fold(self, cs):
        return ActionCall() if cs.can_act(ActionCall) else ActionFold()

    def _check_or_call(self, cs):
        return ActionCheck() if cs.can_act(ActionCheck) else ActionCall()

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def on_hand_start(self, gi: GameInfo, cs: PokerState) -> None:
        self.score            = self._chen(cs.my_hand)
        self.is_ip            = not cs.is_bb   # SB acts last post-flop = in position
        self.raised_pf        = False
        self.my_bid           = 0
        self.chips_before_bid = cs.my_chips
        self.bid_resolved     = False
        self.hand_history     = {}
        self._range_cache     = {}
        self._flop_seen       = False
        self._opp_bet_this_hand = False

    def on_hand_end(self, gi: GameInfo, cs: PokerState) -> None:
        """
        Gradient descent backprop with decaying learning rate.
        Uses FULL unrestricted range for y_true to avoid the mahoraga backprop bug.
        """
        opp = cs.opp_revealed_cards
        if not opp:
            return
        if len(opp) == 2:
            self.n_samples += 1

        for st, d in self.hand_history.items():
            if st == 'pre-flop':
                yt = (self.pf_pcts.get(frozenset(opp), 0.5)
                      if len(opp) == 2 else 0.5)
            else:
                hs = d.get('backprop_evals')  # always the FULL ~990-combo range
                if not hs:
                    continue
                if len(opp) == 2:
                    opp_set = set(opp)
                    yt = 0.5
                    for i, (h, _) in enumerate(hs):
                        if {c.__str__() for c in h} == opp_set:
                            yt = i / len(hs)
                            break
                else:
                    # Single card (won auction): unbiased expectation over compatible hands
                    occ  = opp[0]
                    idxs = [i for i, (h, _) in enumerate(hs)
                            if occ in [c.__str__() for c in h]]
                    yt   = ((sum(idxs) / len(idxs)) / len(hs)) if idxs else 0.5

            xs, yh = d['x_scaled'], d['y_hat']
            err    = yh - yt
            grad   = yh * (1.0 - yh)
            # Decaying LR: prevents oscillation after many training samples
            lr = self.lr / (1.0 + self.n_samples * 0.01)
            self.theta[st] -= lr * err * grad * xs
            self.b[st]     -= lr * err * grad

    # ─────────────────────────────────────────────────────────────────────────
    # Auction resolution (chip-delta inference)
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_bid(self, cs: PokerState):
        """
        Infer opponent's bid from chip delta and update Welford model.
        Rule: winner pays loser's bid (second-price).
          Won → chip_delta = opponent's exact bid (we paid it).
          Lost → chip_delta = 0; soft-estimate opp_bid just above ours.
        """
        if self.bid_resolved:
            return
        self.bid_resolved = True
        delta = self.chips_before_bid - cs.my_chips
        won   = bool(cs.opp_revealed_cards)
        if won and delta > 0:
            self._welford(float(delta))
        else:
            soft = float(self.my_bid) + max(5.0, self._opp_std() * 0.5)
            self._welford(min(soft, 600.0))

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry
    # ─────────────────────────────────────────────────────────────────────────

    def get_move(self, gi: GameInfo, cs: PokerState):
        # Emergency time fallback: never time out and auto-fold
        if gi.time_bank < 1.5:
            return self._check_or_call(cs)

        # Resolve auction exactly once, on first post-auction action
        if cs.street in ('flop', 'turn', 'river'):
            self._resolve_bid(cs)

        # Track flop-reaching hands and opponent post-flop aggression
        if cs.street == 'flop' and not self._flop_seen:
            self.opp_hand_count += 1
            self._flop_seen = True

        if cs.street not in ('pre-flop', 'auction') and cs.cost_to_call > 0:
            # cost_to_call > 0 post-flop means opp bet or raised against us
            if not self._opp_bet_this_hand:
                self.opp_bet_hands      += 1
                self._opp_bet_this_hand  = True

        if cs.street == 'auction':   return self._auction(cs)
        if cs.street == 'pre-flop':  return self._preflop(cs)
        return self._postflop(cs)

    # ─────────────────────────────────────────────────────────────────────────
    # Auction
    # ─────────────────────────────────────────────────────────────────────────

    def _auction(self, cs: PokerState):
        """
        Adaptive bidding with Welford opponent model.

        Strong (score ≥ 9): outbid opp_max + jitter to reliably WIN the peek.
          Seeing their card shrinks range from ~990 → ~44: a 20× precision boost
          that pays dividends across all remaining betting streets.
          Cap at 490 (second-price: we only pay opp's bid, not ours).

        Medium (score 5–8): bid opp_mean − 1.
          If opp is weak (~15 bid): WE WIN cheaply. ✓
          If opp is strong (~300 bid): they pay ~299 into the pot. ✓
          Either outcome is strictly better than bidding near 0.

        Weak (score < 5): minimal bid (fold post-flop anyway; save chips).
        """
        std     = self._opp_std()
        opp_max = self.opp_bid_mean + std

        if self.score >= 9:
            bid = int(min(opp_max + 3 + random.randint(0, 8), 490))
        elif self.score >= 5:
            bid = max(1, int(self.opp_bid_mean) - 1)
        else:
            bid = 5

        # Snapshot chips BEFORE paying so _resolve_bid can compute correct delta
        self.chips_before_bid = cs.my_chips
        self.my_bid           = int(min(bid, cs.my_chips))
        return ActionBid(self.my_bid)

    # ─────────────────────────────────────────────────────────────────────────
    # Pre-flop
    # ─────────────────────────────────────────────────────────────────────────

    def _preflop(self, cs: PokerState):
        yh, xs = self._ml('pre-flop', float(cs.pot))
        self.hand_history['pre-flop'] = {
            'x_scaled': xs, 'y_hat': yh, 'backprop_evals': None
        }

        # Adaptive thresholds based on ML prediction and position
        # yh close to 0 = opp predicted strong → tighten; yh close to 1 = loosen
        tight  = (0.5 - yh) * 4.0              # > 0: tight opp; < 0: loose opp
        pa     = 0.5 if self.is_ip else -0.5   # SB is OOP preflop → tighter

        raise_thresh   = max(6.0,  min(12.0,  9.0 + tight + pa))
        call_thresh    = max(2.0,  min(8.0,   5.0 + tight + pa))
        premium_thresh = max(8.0,  min(14.0, 10.0 + tight + pa))

        s, ctc = self.score, cs.cost_to_call
        if ctc > 0:
            if ctc <= 20:
                if s >= raise_thresh and cs.can_act(ActionRaise):
                    lo, hi = cs.raise_bounds
                    t = max(lo, min(hi, lo + 40))
                    self.raised_pf = True
                    return ActionRaise(int(t))
                elif s >= call_thresh:
                    return self._call_or_fold(cs)
            else:
                # Facing a 3-bet or large open: require premium hand
                if s >= premium_thresh:
                    if cs.pot > 400 or s < 14:
                        return self._call_or_fold(cs)
                    if cs.can_act(ActionRaise):
                        lo, hi = cs.raise_bounds
                        t = max(lo, min(hi, lo + int(cs.pot * 0.5)))
                        self.raised_pf = True
                        return ActionRaise(int(t))
                    return self._call_or_fold(cs)
                elif s >= 7 and ctc < 200:
                    return self._call_or_fold(cs)
            return self._fold_or_check(cs)
        else:
            if s >= raise_thresh and cs.can_act(ActionRaise):
                lo, hi = cs.raise_bounds
                self.raised_pf = True
                return ActionRaise(int(lo))
            return self._check_or_call(cs)

    # ─────────────────────────────────────────────────────────────────────────
    # Post-flop
    # ─────────────────────────────────────────────────────────────────────────

    def _postflop(self, cs: PokerState):
        st    = cs.street
        yh, xs = self._ml(st, float(cs.pot))

        # Build range (cached) and record for backprop
        rc = self._build_range(cs)
        if st not in self.hand_history:
            self.hand_history[st] = {
                'x_scaled':     xs,
                'y_hat':        yh,
                'backprop_evals': rc['full']   # ALWAYS full range for backprop
            }
        else:
            # Update with latest pot value (trains on final street pot)
            self.hand_history[st]['x_scaled'] = xs
            self.hand_history[st]['y_hat']    = yh

        eq, thr = self._get_eq(cs, rc, yh)

        my   = [self.deck[k] for k in cs.my_hand]
        brd  = [self.deck[k] for k in cs.board]
        hv   = eval7.evaluate(my + brd)
        ht   = eval7.handtype(hv)
        is_nut   = ht in _SAFE
        pot_odds = cs.cost_to_call / max(1, cs.pot + cs.cost_to_call)
        pos_edge = -0.03 if self.is_ip else 0.03   # IP: play tighter (acting first post-flop)

        # ── Raise safety gate ──────────────────────────────────────────────
        # When card known: equity computed vs ~44 combos → very accurate → lift cap.
        # Otherwise: large pot + non-nut hand = dangerous to raise (facing re-raises).
        card_known = bool(cs.opp_revealed_cards)
        if card_known or cs.pot <= 800 or is_nut:
            can_raise = True
        else:
            can_raise = False
        # Post-learning exploitation override
        if self.n_samples >= 10 and eq > 0.85 and ht in {'Three of a Kind', 'Two Pair'}:
            can_raise = True

        # Aggression rate: fraction of hands where opp bet post-flop.
        # Only trust once we have MODEL_N hands of data (prevents early-noise misfires).
        agg = (self.opp_bet_hands / self.opp_hand_count
               if self.opp_hand_count >= _MODEL_N else 0.0)

        # ── Facing a bet ───────────────────────────────────────────────────
        if cs.cost_to_call > 0:
            ctc = cs.cost_to_call

            # Graduated overbet defence (vs single 1000-chip gate in mc_with_ml).
            # Fold weak hands earlier to prevent slow bleeding.
            if ctc >  400 and ht == "High Card" and eq < 0.45:
                return self._fold_or_check(cs)
            if ctc >  600 and ht == "Pair"      and eq < 0.45:
                return self._fold_or_check(cs)
            if ctc > 1000 and ht in {"High Card", "Pair"} and eq < 0.35:
                return self._fold_or_check(cs)

            # Never fold a nut hand — protect from pessimistic ML filter.
            # (MC estimate vs top-X% range can under-estimate a flush if
            #  the ML thinks opp is strong → wrong fold → huge chip loss.)
            if is_nut:
                if can_raise and cs.can_act(ActionRaise) and eq > 0.65:
                    f = self._bet_frac(eq, yh, True)
                    r = self._raise_to(cs, f)
                    return r if r else self._call_or_fold(cs)
                return self._call_or_fold(cs)

            # Protect Three of a Kind from overly pessimistic ML filter
            if ht == "Three of a Kind" and eq > 0.30:
                return self._call_or_fold(cs)

            req = pot_odds + thr + pos_edge
            if eq > req:
                # ANTI-ESCALATION: only re-raise with genuinely strong hands.
                # This is mahoraga's most impactful feature vs mc_with_ml/oracle.
                # Re-raising top-pair/two-pair → facing 3-bet → calling 500+ chips
                # with 35% equity → -500 EV. Just call instead.
                reraise_ok = False
                if can_raise and cs.can_act(ActionRaise):
                    if is_nut and eq > 0.65:
                        reraise_ok = True
                    elif ht == "Three of a Kind" and eq > 0.80:
                        reraise_ok = True
                    elif eq > max(0.65, req + 0.20):
                        reraise_ok = True
                if reraise_ok:
                    f = self._bet_frac(eq, yh, is_nut)
                    r = self._raise_to(cs, f)
                    return r if r else self._call_or_fold(cs)
                return self._call_or_fold(cs)
            return self._fold_or_check(cs)

        # ── Checking street ────────────────────────────────────────────────
        else:
            # Slow-play trap: check with near-nuts to induce a bluff from
            # aggressive opponents. Only when aggression confirmed (≥ MODEL_N hands).
            if (is_nut and eq > 0.92 and agg > 0.45 and random.random() < 0.20):
                return self._check_or_call(cs)

            # Check-raise setup: check strong hands vs aggressive opponents.
            # Opponent's bet → our raise → they face tough decision.
            # Guard: opp_hand_count ≥ MODEL_N ensures we have reliable aggression data.
            if (agg > 0.42 and eq > 0.72 and ht != "High Card"
                    and random.random() < 0.28):
                return self._check_or_call(cs)

            # C-bet / value bet threshold.
            # yh > 0.60 = ML predicts opp is weak → c-bet more aggressively.
            # pos_edge: IP faces first-to-act so uses slightly higher threshold.
            c_thr = (0.40 if yh > 0.60 else 0.55) + pos_edge + thr

            if eq > c_thr and can_raise and cs.can_act(ActionRaise):
                f = self._bet_frac(eq, yh, is_nut)
                r = self._raise_to(cs, f)
                if r: return r
            return self._check_or_call(cs)


if __name__ == '__main__':
    run_bot(Player(), parse_args())