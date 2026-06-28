"""AoI-conditioned residual correction on top of kinematic extrapolation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from comm_limited_vmas.estimators.base import BaseEstimator
from comm_limited_vmas.estimators.kinematic import KinematicEstimator


class AoiResidualNetwork(nn.Module):
    """Predicts a bounded acceleration residual and an AoI-conditioned gate."""

    VALID_COVARIANCE_MODES = {"fixed_kinematic", "learned_diag"}

    def __init__(
        self,
        pe_dim: int = 8,
        hidden_layers: Sequence[int] = (128, 128),
        u_max: float = 1.0,
        covariance_mode: str = "fixed_kinematic",
    ):
        super().__init__()
        self.pe_dim = int(pe_dim)
        self.u_max = float(u_max)
        self.covariance_mode = str(covariance_mode)
        if self.covariance_mode not in self.VALID_COVARIANCE_MODES:
            raise ValueError(
                f"Unknown covariance_mode={self.covariance_mode!r}. "
                f"Expected one of {sorted(self.VALID_COVARIANCE_MODES)}"
            )

        self.accel_mlp = make_mlp(
            input_dim=8 + self.pe_dim + 1,
            hidden_layers=hidden_layers,
            output_dim=2,
        )
        self.gate_mlp = make_mlp(
            input_dim=self.pe_dim + 1,
            hidden_layers=hidden_layers,
            output_dim=1,
        )
        self.covariance_mlp = (
            make_mlp(
                input_dim=8 + self.pe_dim + 1,
                hidden_layers=hidden_layers,
                output_dim=4,
            )
            if self.covariance_mode == "learned_diag"
            else None
        )

    def forward(
        self,
        cached_state: Tensor,
        mu_kin: Tensor,
        aoi_steps: Tensor,
        delta_t: Tensor,
        comm_mask: Tensor | None = None,
        kin_cov_diag: Tensor | None = None,
        min_variance: float = 1.0e-4,
        max_variance: float = 10.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        if comm_mask is None:
            comm_mask = torch.zeros_like(aoi_steps, dtype=mu_kin.dtype)
        comm_mask = comm_mask.to(dtype=mu_kin.dtype, device=mu_kin.device)
        pe = positional_encoding(aoi_steps.to(mu_kin.device), self.pe_dim).to(
            dtype=mu_kin.dtype,
            device=mu_kin.device,
        )
        gate_input = torch.cat([pe, comm_mask.unsqueeze(-1)], dim=-1)
        accel_input = torch.cat(
            [mu_kin, cached_state, pe, comm_mask.unsqueeze(-1)],
            dim=-1,
        )

        u_hat = self.u_max * torch.tanh(self.accel_mlp(accel_input))
        rho = torch.sigmoid(self.gate_mlp(gate_input))

        delta_t = delta_t.to(dtype=mu_kin.dtype, device=mu_kin.device).unsqueeze(-1)
        state_delta = torch.cat(
            [0.5 * delta_t.square() * u_hat, delta_t * u_hat],
            dim=-1,
        )
        corrected = mu_kin + rho * state_delta

        covariance_diag = None
        if kin_cov_diag is not None:
            covariance_diag = self.self_covariance_diag(
                accel_input=accel_input,
                kin_cov_diag=kin_cov_diag,
                min_variance=min_variance,
                max_variance=max_variance,
            )
        return corrected, u_hat, rho, covariance_diag

    def self_covariance_diag(
        self,
        accel_input: Tensor,
        kin_cov_diag: Tensor,
        min_variance: float,
        max_variance: float,
    ) -> Tensor:
        kin_cov_diag = kin_cov_diag.to(
            device=accel_input.device,
            dtype=accel_input.dtype,
        )
        if self.covariance_mlp is None:
            residual_var = torch.zeros_like(kin_cov_diag)
        else:
            residual_var = F.softplus(self.covariance_mlp(accel_input)) + float(
                min_variance
            )
        return (kin_cov_diag + residual_var).clamp(
            min=float(min_variance),
            max=float(max_variance),
        )


class AoiResidualEstimator(BaseEstimator):
    """Frozen residual estimator used as a drop-in communication state estimator."""

    def __init__(
        self,
        state_dim: int,
        device: torch.device,
        dt: float = 0.1,
        sigma_p: float = 0.01,
        sigma_v: float = 0.01,
        process_noise_q: float = 1.0,
        max_extrapolation_steps: int | None = 15,
        pe_dim: int = 8,
        hidden_layers: Sequence[int] = (128, 128),
        u_max: float = 1.0,
        covariance_mode: str = "fixed_kinematic",
        min_variance: float = 1.0e-4,
        max_variance: float = 10.0,
        checkpoint_path: str | None = None,
        fallback_to_kinematic: bool = True,
        **_: object,
    ):
        super().__init__(state_dim=state_dim, device=device)
        self.kinematic = KinematicEstimator(
            state_dim=state_dim,
            device=device,
            dt=dt,
            sigma_p=sigma_p,
            sigma_v=sigma_v,
            process_noise_q=process_noise_q,
            max_extrapolation_steps=max_extrapolation_steps,
        )
        self.network = AoiResidualNetwork(
            pe_dim=pe_dim,
            hidden_layers=hidden_layers,
            u_max=u_max,
            covariance_mode=covariance_mode,
        ).to(device)
        self.min_variance = float(min_variance)
        self.max_variance = float(max_variance)
        self.has_checkpoint = False
        self.fallback_to_kinematic = bool(fallback_to_kinematic)

        if checkpoint_path:
            self.load_checkpoint(Path(checkpoint_path))
        elif not self.fallback_to_kinematic:
            raise FileNotFoundError(
                "aoi_residual requires checkpoint_path when fallback_to_kinematic=false"
            )

        self.network.eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)

    def estimate_state(
        self,
        cached_state: Tensor,
        aoi_steps: Tensor,
        comm_mask: Tensor | None = None,
    ) -> Tensor:
        mean, _ = self.estimate_distribution(cached_state, aoi_steps, comm_mask)
        return mean

    def estimate_distribution(
        self,
        cached_state: Tensor,
        aoi_steps: Tensor,
        comm_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        mu_kin = self.kinematic.estimate_state(cached_state, aoi_steps, comm_mask)
        kin_cov_diag = self.kinematic.covariance_diag(aoi_steps).to(
            device=mu_kin.device,
            dtype=mu_kin.dtype,
        )
        if not self.has_checkpoint:
            return mu_kin, kin_cov_diag

        delta_t = self.kinematic._delta_t(aoi_steps, clamp=True)
        with torch.no_grad():
            corrected, _, _, covariance_diag = self.network(
                cached_state=cached_state,
                mu_kin=mu_kin,
                aoi_steps=aoi_steps,
                delta_t=delta_t,
                comm_mask=comm_mask,
                kin_cov_diag=kin_cov_diag,
                min_variance=self.min_variance,
                max_variance=self.max_variance,
            )
        assert covariance_diag is not None
        return corrected, covariance_diag

    def covariance_diag(self, aoi_steps: Tensor) -> Tensor:
        return self.kinematic.covariance_diag(aoi_steps)

    def estimate_receiver_states(
        self,
        cached_states: Tensor,
        aoi_steps: Tensor,
        comm_mask: Tensor | None = None,
        receiver: int | None = None,
    ) -> Tensor:
        mean, _ = self.estimate_distribution(cached_states, aoi_steps, comm_mask)
        return mean

    def estimate_receiver_distribution(
        self,
        cached_states: Tensor,
        aoi_steps: Tensor,
        comm_mask: Tensor | None = None,
        receiver: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self.estimate_distribution(cached_states, aoi_steps, comm_mask)

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint_path = checkpoint_path.expanduser()
        if not checkpoint_path.exists():
            if self.fallback_to_kinematic:
                return
            raise FileNotFoundError(f"AoI residual checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        try:
            self.network.load_state_dict(state_dict)
        except RuntimeError as err:
            raise RuntimeError(
                "Failed to load AoI residual checkpoint. Check that the estimator "
                "covariance_mode matches the checkpoint; NLL learned_diag models "
                "must be retrained with the covariance head."
            ) from err
        self.has_checkpoint = True


def make_mlp(
    input_dim: int,
    hidden_layers: Sequence[int],
    output_dim: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(nn.Tanh())
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


def positional_encoding(aoi_steps: Tensor, pe_dim: int) -> Tensor:
    if pe_dim <= 0:
        return aoi_steps.new_zeros((*aoi_steps.shape, 0), dtype=torch.float32)

    half_dim = max(1, pe_dim // 2)
    frequencies = torch.exp(
        torch.linspace(
            0.0,
            torch.log(torch.tensor(1000.0, device=aoi_steps.device)),
            half_dim,
            device=aoi_steps.device,
        )
    )
    angles = aoi_steps.float().unsqueeze(-1) / frequencies
    pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if pe.shape[-1] > pe_dim:
        pe = pe[..., :pe_dim]
    elif pe.shape[-1] < pe_dim:
        pe = torch.nn.functional.pad(pe, (0, pe_dim - pe.shape[-1]))
    return pe
