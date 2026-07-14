###
# Streaming-tap variant of RecursiveJEPAFeedbackEncoder.
#
# forward() is inherited UNCHANGED: still only the true/eval_horizon "final"
# frame produces a decodable pyramid, exactly as before -- so this class is
# still a drop-in for the evaluator / BeliefEvolutionTracker.
#
# The addition is forward_stream(): a GENERATOR that walks the same
# recurrence ONCE, start to finish, and yields a decodable tap list at EVERY
# frame along the way (not just the last one). This is what makes
# "backprop the decoder at every frame, update, then continue" cheap: a
# single O(T) pass, instead of the O(T^2) that re-deriving a tap at frame h
# by replaying frames 0..h from scratch (the eval_horizon trick used by
# trainer_all_horizons.py) would cost over h = 0..T-1.
#
# To get a REAL tap at every frame (not just the propagate_outer_steps
# shortcut the parent uses for non-final frames), every frame here runs the
# full n_outer-cycle loop -- exactly what the parent's "final frame" branch
# does. Under your actual configs (n_outer_steps=1, propagate_outer_steps=1)
# this is free (they're already the same number of cycles); for configs
# where n_outer_steps > propagate_outer_steps it's genuinely more expensive
# per frame than plain forward(), but still O(T) overall, not O(T^2).
#
# tap_source="outer" snapshots z_H after each outer cycle (n_taps = n_outer).
# tap_source="layers" unrolls H_block into its internal GatedBlocks on the
# LAST outer cycle of EVERY frame and snapshots after each internal layer
# (n_taps = n_layers_per_stream), mirroring the parent forward()'s
# final-frame semantics at every step of the stream. The unroll costs
# nothing extra (the blocks run anyway; we just keep the intermediates).



import torch

from pangaea.encoders.recursive_jepa_feedback_encoder import RecursiveJEPAFeedbackEncoder


class RecursiveJEPAFeedbackStreamEncoder(RecursiveJEPAFeedbackEncoder):
    """RecursiveJEPAFeedbackEncoder + forward_stream(), a generator yielding
    a decodable feature pyramid at every frame of a single continuous pass
    through the recurrence. Always runs with the backbone frozen (no_grad) --
    not intended for finetune=true."""

    def forward_stream(self, image: dict[str, torch.Tensor], min_horizon: int = 0):
        """Yields (t, taps) for t = min_horizon .. T-1, taps being a
        list[Tensor] in the exact format forward() returns (ready to feed
        into the decoder's neck/_forward_feature/dropout/conv_seg pipeline).

        The caller is expected to consume+decode+backprop+step between
        yields; z_H/z_L continue evolving from wherever they were left
        (nothing is recomputed), matching a single ordinary forward() pass's
        total cost (see the full-loop note in the module header).
        """
        x = image["optical"]                      # (B, C, T, H, W)
        B, C, T, H, W = x.shape
        pos = self.model.pos_embed

        z_L = self.model.zL_init.expand(B, self.num_patches, -1).to(x.device).contiguous()
        z_H = None

        for t in range(T):
            # Scoped to close BEFORE the yield below: a `with` block left
            # open across a `yield` stays open in the CALLER's code too
            # (generators suspend mid-block, they don't exit it), which
            # would silently disable grad in the caller's own decode +
            # backward step between iterations. Keeping no_grad strictly
            # per-frame avoids that trap.
            with torch.no_grad():
                P_t = self.model.patch_embed(x[:, :, t]) + pos
                z_H = self._fold(z_H, P_t)

                taps: list[torch.Tensor] = []
                for c in range(self.n_outer):
                    ctx = self._inner_context(z_H, P_t)
                    for _ in range(self.n_inner):
                        z_L = self.model.L_block(z_L + ctx)
                    if self.tap_source == "outer":
                        z_H = self.model.H_block(z_H + z_L)
                        taps.append(z_H)
                    elif c < self.n_outer - 1:
                        z_H = self.model.H_block(z_H + z_L)
                    else:
                        # LAST outer cycle: unroll H_block into its internal
                        # GatedBlocks, snapshot after each (parent forward()'s
                        # final-frame "layers" semantics, applied per frame).
                        h = z_H + z_L
                        for blk in self.model.H_block.blocks:
                            h = blk(h)
                            taps.append(h)
                        z_H = self.model.H_block.norm_f(h)

                outputs: list[torch.Tensor] | None = None
                if t >= min_horizon:
                    outputs = []
                    for layer in self.output_layers:
                        feat = self.model.final_norm(taps[layer])    # (B, L, D)
                        feat = feat.transpose(1, 2).reshape(
                            B, self.embed_dim, self.grid_size, self.grid_size
                        ).contiguous()                                # (B, D, 8, 8)
                        outputs.append(feat)

            if outputs is not None:
                yield t, outputs
