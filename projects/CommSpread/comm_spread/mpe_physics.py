"""MPE-compatible VMAS physics primitives used by the strict SAR profile."""

from __future__ import annotations

import torch
from torch import Tensor
from vmas.simulator.core import Entity, World


class MPEParityWorld(World):
    """World with the original MPE soft-contact response.

    VMAS normally rejects non-overlapping pairs before contact evaluation and
    truncates repulsion to zero outside the summed radii. MPE evaluates its
    softplus contact for every collidable pair, including the exponentially
    small tail immediately outside contact. With only three SAR spheres, doing
    the same has negligible overhead and makes collision probes comparable.
    """

    def collides(self, a: Entity, b: Entity) -> bool:
        if (not a.collides(b)) or (not b.collides(a)) or a is b:
            return False
        if not a.movable and not a.rotatable and not b.movable and not b.rotatable:
            return False
        return {a.shape.__class__, b.shape.__class__} in self._collidable_pairs

    def _get_constraint_forces(
        self,
        pos_a: Tensor,
        pos_b: Tensor,
        dist_min,
        force_multiplier: float,
        attractive: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if attractive:
            return super()._get_constraint_forces(
                pos_a,
                pos_b,
                dist_min,
                force_multiplier,
                attractive=True,
            )

        delta_pos = pos_a - pos_b
        dist = torch.linalg.vector_norm(delta_pos, dim=-1)
        margin = self._contact_margin
        penetration = torch.logaddexp(
            torch.zeros((), dtype=torch.float32, device=self.device),
            (dist_min - dist) / margin,
        ) * margin
        direction = delta_pos / torch.where(dist > 0, dist, 1.0).unsqueeze(-1)
        force = force_multiplier * direction * penetration.unsqueeze(-1)
        # MPE is undefined for coincident centers. Use the permutation-neutral
        # zero vector instead of introducing an arbitrary separating direction.
        force = torch.where((dist > 0).unsqueeze(-1), force, torch.zeros_like(force))
        return force, -force
