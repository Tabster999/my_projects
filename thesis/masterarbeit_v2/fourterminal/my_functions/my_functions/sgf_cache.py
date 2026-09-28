"""
Surface Green's function cache.

Sancho-Rubio decimation depends ONLY on (H_onsite, V_hop, z) -- the coupling
block V_coupling enters afterwards, in self_energy().  In the 4-terminal
geometry several leads share the same (H_onsite, V_hop):

  * lead_L and lead_R are built from the SAME H_layer_N and V_n
    (junction.py:_build).  They differ only in V_coupling / br.
    -> identical surface GF, currently decimated twice.

  * ribbon_bot (phase 0) and ribbon_top at phi_ref=0 are built from the
    same onsite_SC/H_intra/H_inter; they differ only in tc_bot vs tc_top.
    -> identical surface GF whenever phi_ref == 0 (the usual choice),
       currently decimated twice.

Profiling at nx=12, ny=20, N_E=41: surface_gf was 3.09 s of 5.05 s total
setup (61%), so removing the two duplicates is the single cheapest win
available.

Usage: wrap a Lead so .surface_gf(z) hits the cache.
"""

import numpy as np


class SurfaceGFCache:
    """Memoize Sancho-Rubio results across leads that share (H_onsite, V_hop)."""

    def __init__(self):
        self._store = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(lead, z_batch):
        # z_batch is z * I, so its diagonal is the energy grid; hash that plus
        # the two matrices that actually determine the decimation.
        zdiag = np.einsum('nii->ni', z_batch)
        return (lead.H_onsite.tobytes(), lead.V_hop.tobytes(),
                zdiag.tobytes(), lead.p.max_iter, lead.p.tol)

    def surface_gf(self, lead, z_batch):
        k = self._key(lead, z_batch)
        if k in self._store:
            self.hits += 1
            return self._store[k]
        self.misses += 1
        g = lead.surface_gf(z_batch)
        self._store[k] = g
        return g

    def self_energy(self, lead, z_batch):
        """Same contract as Lead.self_energy, but reuses a cached surface GF."""
        g = self.surface_gf(lead, z_batch)
        Vc = lead.V_coupling[None, :, :]
        return Vc.conj().transpose(0, 2, 1) @ g @ Vc

    def clear(self):
        self._store.clear()