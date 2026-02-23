"""
Diapedesis GPU Kernels (NVIDIA Warp)
=====================================

Warp kernels for GPU-accelerated diapedesis simulation:
- Poiseuille flow profile in cylindrical vessel
- Stochastic selectin bind/unbind
- Integrin adhesion forces
- Transmigration force through vessel wall
- Vessel boundary enforcement

These kernels parallelize the per-particle force computations
that would otherwise be O(N) serial loops in the CPU path.
Falls back gracefully to NumPy when Warp is unavailable.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

try:
    from .warp_backend import check_warp_available
    _WARP_AVAILABLE = check_warp_available()
except ImportError:
    _WARP_AVAILABLE = False

if _WARP_AVAILABLE:
    import warp as wp

    # ── Shear flow kernel ──────────────────────────────────────────────

    @wp.kernel
    def shear_flow_kernel(
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        forces: wp.array(dtype=wp.vec3),
        vessel_radius: wp.float32,
        max_velocity: wp.float32,
        damping: wp.float32,
        n_particles: wp.int32,
    ):
        """Poiseuille parabolic velocity profile in cylindrical vessel.

        v(r) = v_max * (1 - (r/R)²) along x-axis.
        Applies drag force: F = damping * (v_target - v_current).
        """
        i = wp.tid()
        if i >= n_particles:
            return

        pos = positions[i]
        vel = velocities[i]

        # Radial distance from vessel axis (y=0, z=0)
        r = wp.sqrt(pos[1] * pos[1] + pos[2] * pos[2])
        ratio = r / vessel_radius
        if ratio > 1.0:
            ratio = 1.0

        # Parabolic profile
        v_flow = max_velocity * (1.0 - ratio * ratio)

        # Drag toward flow velocity (x direction only)
        fx = damping * (v_flow - vel[0])

        f_old = forces[i]
        forces[i] = wp.vec3(f_old[0] + fx, f_old[1], f_old[2])

    # ── Brownian force kernel ──────────────────────────────────────────

    @wp.kernel
    def brownian_force_kernel(
        forces: wp.array(dtype=wp.vec3),
        noise: wp.array(dtype=wp.vec3),
        strength: wp.float32,
        n_particles: wp.int32,
    ):
        """Add Brownian (thermal) noise force."""
        i = wp.tid()
        if i >= n_particles:
            return

        f = forces[i]
        n = noise[i]
        forces[i] = wp.vec3(
            f[0] + strength * n[0],
            f[1] + strength * n[1],
            f[2] + strength * n[2],
        )

    # ── Selectin binding kernel ────────────────────────────────────────

    @wp.kernel
    def selectin_rolling_force_kernel(
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        forces: wp.array(dtype=wp.vec3),
        selectin_bonds: wp.array(dtype=wp.int32),
        vessel_radius: wp.float32,
        bond_force: wp.float32,
        rolling_velocity: wp.float32,
        damping: wp.float32,
        n_leukocytes: wp.int32,
    ):
        """Apply selectin-mediated rolling forces on bound leukocytes.

        When selectin-bound:
        - Radial force pulling toward wall
        - Tangential drag to reduce speed to rolling velocity
        """
        i = wp.tid()
        if i >= n_leukocytes:
            return

        if selectin_bonds[i] == 0:
            return

        pos = positions[i]
        vel = velocities[i]

        # Radial direction (outward from axis toward wall)
        yz_dist = wp.sqrt(pos[1] * pos[1] + pos[2] * pos[2])
        if yz_dist < 1.0e-6:
            return

        # Wall-ward force (pull toward endothelium)
        norm_y = pos[1] / yz_dist
        norm_z = pos[2] / yz_dist
        fy_wall = norm_y * bond_force * 0.1
        fz_wall = norm_z * bond_force * 0.1

        # Rolling: drag velocity toward rolling_velocity in x
        v_flow_capped = rolling_velocity
        if vel[0] > rolling_velocity:
            v_flow_capped = rolling_velocity
        fx_roll = damping * 2.0 * (v_flow_capped - vel[0])

        f_old = forces[i]
        forces[i] = wp.vec3(
            f_old[0] + fx_roll,
            f_old[1] + fy_wall,
            f_old[2] + fz_wall,
        )

    # ── Integrin adhesion kernel ───────────────────────────────────────

    @wp.kernel
    def integrin_adhesion_kernel(
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        forces: wp.array(dtype=wp.vec3),
        leukocyte_states: wp.array(dtype=wp.int32),
        integrin_activation: wp.array(dtype=wp.float32),
        vessel_radius: wp.float32,
        bond_strength: wp.float32,
        damping: wp.float32,
        n_leukocytes: wp.int32,
    ):
        """Strong adhesion force for activated integrins.

        When integrin activation > 0.5 and state is ACTIVATING/ARRESTED/CRAWLING:
        - Spring force toward wall contact point
        - Strong drag opposing flow (firm arrest)
        """
        i = wp.tid()
        if i >= n_leukocytes:
            return

        activation = integrin_activation[i]
        state = leukocyte_states[i]

        # States 3=ACTIVATING, 4=ARRESTED, 5=CRAWLING
        if activation < 0.5:
            return
        if state < 3 or state > 5:
            return

        pos = positions[i]
        vel = velocities[i]

        # Radial direction toward wall
        yz_dist = wp.sqrt(pos[1] * pos[1] + pos[2] * pos[2])
        if yz_dist < 1.0e-6:
            return

        norm_y = pos[1] / yz_dist
        norm_z = pos[2] / yz_dist

        # Adhesion toward wall
        adhesion_f = activation * bond_strength * 0.05
        fy = norm_y * adhesion_f
        fz = norm_z * adhesion_f

        # Arrest: strong drag opposing flow
        fx = damping * 5.0 * (0.0 - vel[0])

        f_old = forces[i]
        forces[i] = wp.vec3(f_old[0] + fx, f_old[1] + fy, f_old[2] + fz)

    # ── Transmigration kernel ──────────────────────────────────────────

    @wp.kernel
    def transmigration_kernel(
        positions: wp.array(dtype=wp.vec3),
        forces: wp.array(dtype=wp.vec3),
        leukocyte_states: wp.array(dtype=wp.int32),
        transmigration_progress: wp.array(dtype=wp.float32),
        vessel_radius: wp.float32,
        transmigration_speed: wp.float32,
        damping: wp.float32,
        n_leukocytes: wp.int32,
    ):
        """Push transmigrating leukocytes radially outward through wall.

        Only applies to state 6 (TRANSMIGRATING).
        """
        i = wp.tid()
        if i >= n_leukocytes:
            return

        if leukocyte_states[i] != 6:  # TRANSMIGRATING
            return

        pos = positions[i]
        yz_dist = wp.sqrt(pos[1] * pos[1] + pos[2] * pos[2])
        if yz_dist < 1.0e-6:
            return

        # Outward direction
        speed = transmigration_speed / 60.0  # μm/s from μm/min
        norm_y = pos[1] / yz_dist
        norm_z = pos[2] / yz_dist

        fy = norm_y * speed * damping
        fz = norm_z * speed * damping

        f_old = forces[i]
        forces[i] = wp.vec3(f_old[0], f_old[1] + fy, f_old[2] + fz)

    # ── Boundary kernel ────────────────────────────────────────────────

    @wp.kernel
    def vessel_boundary_kernel(
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        particle_types: wp.array(dtype=wp.int32),
        leukocyte_states: wp.array(dtype=wp.int32),
        radii: wp.array(dtype=wp.float32),
        vessel_radius: wp.float32,
        vessel_length: wp.float32,
        n_leukocytes: wp.int32,
        n_total: wp.int32,
    ):
        """Enforce vessel boundaries.

        - RBCs and flowing/rolling leukocytes: stay inside vessel
        - Migrated leukocytes: stay outside vessel
        - X-axis: periodic wrapping
        """
        i = wp.tid()
        if i >= n_total:
            return

        pos = positions[i]
        vel = velocities[i]
        R = vessel_radius

        # X periodic
        x = pos[0]
        if x < 0.0:
            x = x + vessel_length
        elif x > vessel_length:
            x = x - vessel_length

        y = pos[1]
        z = pos[2]

        r = wp.sqrt(y * y + z * z)
        ptype = particle_types[i]

        if ptype == 1:  # RBC
            max_r = R - radii[i]
            if r > max_r and r > 1.0e-6:
                scale = max_r / r
                y = y * scale
                z = z * scale
                vel = wp.vec3(vel[0], vel[1] * (-0.3), vel[2] * (-0.3))

        elif ptype == 0 and i < n_leukocytes:  # Leukocyte
            state = leukocyte_states[i]
            if state <= 5:  # up to CRAWLING
                max_r = R - radii[i] * 0.3
                if r > max_r and r > 1.0e-6:
                    scale = max_r / r
                    y = y * scale
                    z = z * scale
            elif state == 7:  # MIGRATED
                min_r = R + radii[i]
                if r < min_r and r > 1.0e-6:
                    scale = min_r / r
                    y = y * scale
                    z = z * scale

        positions[i] = wp.vec3(x, y, z)
        velocities[i] = vel

    # ── Integration kernel ─────────────────────────────────────────────

    @wp.kernel
    def integrate_overdamped_kernel(
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        forces: wp.array(dtype=wp.vec3),
        damping: wp.float32,
        max_velocity: wp.float32,
        dt: wp.float32,
        n_particles: wp.int32,
    ):
        """Overdamped Euler integration: v = F/γ, x += v*dt.

        Includes velocity clamping for numerical stability.
        """
        i = wp.tid()
        if i >= n_particles:
            return

        f = forces[i]
        inv_damp = 1.0 / damping

        vx = f[0] * inv_damp
        vy = f[1] * inv_damp
        vz = f[2] * inv_damp

        # Clamp velocity
        speed = wp.sqrt(vx * vx + vy * vy + vz * vz)
        if speed > max_velocity:
            scale = max_velocity / speed
            vx = vx * scale
            vy = vy * scale
            vz = vz * scale

        velocities[i] = wp.vec3(vx, vy, vz)

        pos = positions[i]
        positions[i] = wp.vec3(
            pos[0] + vx * dt,
            pos[1] + vy * dt,
            pos[2] + vz * dt,
        )


# ── GPU Accelerator Class ─────────────────────────────────────────────

class DiapedesisGPU:
    """GPU-accelerated force computation for DiapedesisSim.

    Wraps Warp kernels with NumPy CPU fallback. The DiapedesisSim engine
    calls this class's methods instead of its own per-particle loops
    when GPU is available.

    Usage::

        gpu = DiapedesisGPU(device="cuda:0")
        if gpu.available:
            gpu.upload(positions, velocities, forces, ...)
            gpu.apply_shear_flow(cfg)
            gpu.apply_brownian(cfg, rng)
            gpu.integrate(cfg)
            gpu.download(positions, velocities)
    """

    def __init__(self, device: str = None):
        self.available = _WARP_AVAILABLE
        self.device = device

        if self.available:
            import warp as wp
            if device is None:
                self.device = "cuda:0" if wp.is_cuda_available() else "cpu"
            wp.init()
            log.info(f"DiapedesisGPU initialized on {self.device}")
        else:
            log.info("DiapedesisGPU: Warp unavailable, using CPU fallback")

        # Device arrays (allocated on upload)
        self._positions = None
        self._velocities = None
        self._forces = None
        self._particle_types = None
        self._radii = None
        self._selectin_bonds = None
        self._leukocyte_states = None
        self._integrin_activation = None
        self._transmigration_progress = None
        self._noise = None
        self._n_total = 0
        self._n_leukocytes = 0

    def upload(self, positions: np.ndarray, velocities: np.ndarray,
               forces: np.ndarray, particle_types: np.ndarray,
               radii: np.ndarray, selectin_bonds: np.ndarray,
               leukocyte_states: np.ndarray, integrin_activation: np.ndarray,
               transmigration_progress: np.ndarray,
               n_leukocytes: int):
        """Upload arrays to GPU."""
        if not self.available:
            return

        self._n_total = len(positions)
        self._n_leukocytes = n_leukocytes

        # Convert to vec3 for position arrays
        pos_v3 = np.ascontiguousarray(positions, dtype=np.float32)
        vel_v3 = np.ascontiguousarray(velocities, dtype=np.float32)

        self._positions = wp.array(pos_v3, dtype=wp.vec3, device=self.device)
        self._velocities = wp.array(vel_v3, dtype=wp.vec3, device=self.device)
        self._forces = wp.zeros(self._n_total, dtype=wp.vec3, device=self.device)
        self._particle_types = wp.array(
            particle_types.astype(np.int32), dtype=wp.int32, device=self.device)
        self._radii = wp.array(
            radii.astype(np.float32), dtype=wp.float32, device=self.device)
        self._selectin_bonds = wp.array(
            selectin_bonds.astype(np.int32), dtype=wp.int32, device=self.device)
        self._leukocyte_states = wp.array(
            leukocyte_states.astype(np.int32), dtype=wp.int32, device=self.device)
        self._integrin_activation = wp.array(
            integrin_activation.astype(np.float32), dtype=wp.float32, device=self.device)
        self._transmigration_progress = wp.array(
            transmigration_progress.astype(np.float32), dtype=wp.float32, device=self.device)

    def zero_forces(self):
        """Zero the force array on GPU."""
        if not self.available:
            return
        self._forces.zero_()

    def apply_shear_flow(self, vessel_radius: float, max_velocity: float,
                          damping: float):
        """Launch shear flow kernel."""
        if not self.available:
            return
        wp.launch(
            shear_flow_kernel,
            dim=self._n_total,
            inputs=[
                self._positions, self._velocities, self._forces,
                float(vessel_radius), float(max_velocity), float(damping),
                self._n_total,
            ],
            device=self.device,
        )

    def apply_brownian(self, strength: float, rng: np.random.Generator):
        """Launch Brownian force kernel."""
        if not self.available:
            return
        noise_np = rng.standard_normal((self._n_total, 3)).astype(np.float32)
        noise_wp = wp.array(noise_np, dtype=wp.vec3, device=self.device)
        wp.launch(
            brownian_force_kernel,
            dim=self._n_total,
            inputs=[self._forces, noise_wp, float(strength), self._n_total],
            device=self.device,
        )

    def apply_selectin_rolling(self, vessel_radius: float, bond_force: float,
                                rolling_velocity: float, damping: float):
        """Launch selectin rolling force kernel."""
        if not self.available:
            return
        wp.launch(
            selectin_rolling_force_kernel,
            dim=self._n_leukocytes,
            inputs=[
                self._positions, self._velocities, self._forces,
                self._selectin_bonds,
                float(vessel_radius), float(bond_force),
                float(rolling_velocity), float(damping),
                self._n_leukocytes,
            ],
            device=self.device,
        )

    def apply_integrin_adhesion(self, vessel_radius: float, bond_strength: float,
                                 damping: float):
        """Launch integrin adhesion kernel."""
        if not self.available:
            return
        wp.launch(
            integrin_adhesion_kernel,
            dim=self._n_leukocytes,
            inputs=[
                self._positions, self._velocities, self._forces,
                self._leukocyte_states, self._integrin_activation,
                float(vessel_radius), float(bond_strength), float(damping),
                self._n_leukocytes,
            ],
            device=self.device,
        )

    def apply_transmigration(self, vessel_radius: float,
                              transmigration_speed: float, damping: float):
        """Launch transmigration kernel."""
        if not self.available:
            return
        wp.launch(
            transmigration_kernel,
            dim=self._n_leukocytes,
            inputs=[
                self._positions, self._forces,
                self._leukocyte_states, self._transmigration_progress,
                float(vessel_radius), float(transmigration_speed), float(damping),
                self._n_leukocytes,
            ],
            device=self.device,
        )

    def integrate(self, damping: float, max_velocity: float, dt: float):
        """Launch integration kernel."""
        if not self.available:
            return
        wp.launch(
            integrate_overdamped_kernel,
            dim=self._n_total,
            inputs=[
                self._positions, self._velocities, self._forces,
                float(damping), float(max_velocity), float(dt),
                self._n_total,
            ],
            device=self.device,
        )

    def apply_boundary(self, vessel_radius: float, vessel_length: float):
        """Launch boundary enforcement kernel."""
        if not self.available:
            return
        wp.launch(
            vessel_boundary_kernel,
            dim=self._n_total,
            inputs=[
                self._positions, self._velocities,
                self._particle_types, self._leukocyte_states,
                self._radii,
                float(vessel_radius), float(vessel_length),
                self._n_leukocytes, self._n_total,
            ],
            device=self.device,
        )

    def download(self, positions: np.ndarray, velocities: np.ndarray):
        """Download results from GPU back to NumPy arrays."""
        if not self.available:
            return
        positions[:] = self._positions.numpy().reshape(-1, 3)
        velocities[:] = self._velocities.numpy().reshape(-1, 3)

    def sync(self):
        """Synchronize GPU stream."""
        if self.available:
            wp.synchronize()
