###
# Δt-conditioned variant of RecursiveJEPAFeedbackEncoder.
#
# Everything about the recurrence (z_L worker state, z_H belief/feedback, inner/
# outer loops, final-frame pyramid taps, truncated BPTT) is inherited unchanged
# from the parent. The ONLY difference is the temporal FOLD that combines the
# running belief with each new frame's evidence: instead of a constant `alpha`
# (or a plain sum), the mixing weight is a function of the REAL time gap between
# consecutive Sentinel-2 acquisitions:
#
#     Δt_t    = date_t - date_{t-1}            (days, >= 0 for a sorted series)
#     alpha_t = exp(-Δt_t / tau)              (in (0, 1])
#     z_H_t   = alpha_t * z_H_{t-1} + (1 - alpha_t) * P_t
#
# A long gap (Δt >> tau) -> alpha_t -> 0 -> the stale belief is forgotten and the
# new frame dominates; a short gap (Δt << tau) -> alpha_t -> 1 -> the belief is
# retained and the (redundant) new frame barely perturbs it. This is a
# continuous-time leaky integrator keyed to the irregular sampling -- an
# aggregation that a decoder-side temporal pool structurally cannot express,
# which is the whole point of keeping the aggregator inside the (frozen) encoder.
#
# Dates arrive via image["optical_dates"] (shape (B, T), integer days since the
# dataset reference_date; produced by PastisS2Dates and lifted from metadata into
# the image dict by collate_fn). When dates are ABSENT the fold transparently
# falls back to the parent's constant behaviour, so this stays a drop-in
# replacement for RecursiveJEPAFeedbackEncoder.
###

import torch

from pangaea.encoders.recursive_jepa_feedback_encoder import RecursiveJEPAFeedbackEncoder


class RecursiveJEPAFeedbackDtEncoder(RecursiveJEPAFeedbackEncoder):
    """RecursiveJEPAFeedbackEncoder with a Δt-conditioned temporal fold."""

    def __init__(self, *args, tau: float = 30.0, dt_fold: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        if tau <= 0:
            raise ValueError(f"tau must be > 0 (days), got {tau}.")
        # time constant of the exponential forgetting, in the same units as the
        # dates (days since reference_date, as returned by prepare_dates).
        self.tau = float(tau)
        # master switch: if False, behaves exactly like the parent (constant fold).
        self.dt_fold = bool(dt_fold)
        # per-forward state: a per-frame alpha schedule and a cursor that _fold
        # consumes. The parent's forward calls _fold exactly once per frame, in
        # temporal order t = 0, 1, ..., last -- so a simple cursor stays aligned.
        self._alpha_sched: list[torch.Tensor | None] | None = None
        self._fold_cursor: int = 0

    def _setup_alpha_schedule(self, image: dict[str, torch.Tensor]) -> None:
        """Build the per-frame alpha schedule from image['optical_dates'] and
        reset the fold cursor. Called by EVERY entry point that drives the
        recurrence (forward AND the belief_cosines diagnostic), because _fold
        consumes the cursor in temporal order and would otherwise read stale
        state left over from a previous call."""
        x = image["optical"]                       # (B, C, T, H, W)
        B, _, T, _, _ = x.shape
        dates = image.get("optical_dates", None)   # (B, T) or None

        if self.dt_fold and dates is not None and T > 1:
            dates = dates.to(device=x.device, dtype=torch.float32)   # (B, T)
            dt = (dates[:, 1:] - dates[:, :-1]).clamp(min=0.0)        # (B, T-1)
            alpha = torch.exp(-dt / self.tau)                        # (B, T-1)
            # alpha for frame t (t >= 1) is column t-1; the seed frame (t=0) never
            # reads the schedule because _fold returns P_t when z_H is None.
            self._alpha_sched = [None] + [
                alpha[:, t - 1].view(B, 1, 1) for t in range(1, T)
            ]
        else:
            self._alpha_sched = None                 # -> parent constant fold

        self._fold_cursor = 0

    def forward(self, image: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        self._setup_alpha_schedule(image)
        return super().forward(image)

    @torch.no_grad()
    def belief_cosines(self, image: dict[str, torch.Tensor]) -> list[float]:
        # belief_cosines mirrors forward's recurrence but does NOT call forward(),
        # so set up the same schedule (and reset the cursor) here, else _fold reads
        # a stale cursor from the previous forward() -> IndexError.
        self._setup_alpha_schedule(image)
        try:
            return super().belief_cosines(image)
        finally:
            self._alpha_sched = None
            self._fold_cursor = 0

    def _fold(
        self, z_H: torch.Tensor | None, P_t: torch.Tensor
    ) -> torch.Tensor:
        t = self._fold_cursor
        self._fold_cursor += 1

        if z_H is None:                              # t == 0: seed belief
            return P_t
        # No schedule (no dates), or a cursor that has run past it (a caller that
        # bypassed _setup_alpha_schedule): fall back to the parent constant fold
        # rather than crash.
        if self._alpha_sched is None or t >= len(self._alpha_sched):
            return super()._fold(z_H, P_t)

        a = self._alpha_sched[t].to(device=P_t.device, dtype=P_t.dtype)  # (B,1,1)
        return a * z_H + (1.0 - a) * P_t
