"""Unit tests for dual-stream condition construction."""

import unittest

import torch

from acestep.training_v2.dual_stream import (
    MOTIF_ROLE,
    VOCAL_ROLE,
    DualStreamConditioner,
    masked_adaptive_pool,
)


class DualStreamConditionerTests(unittest.TestCase):
    """Validate shape and masking behavior without loading ACE-Step weights."""

    def test_masked_pool_ignores_padded_values(self):
        """Padding must not change pooled values."""
        latents = torch.tensor([[[2.0], [4.0], [100.0], [100.0]]])
        mask = torch.tensor([[1, 1, 0, 0]])
        pooled, pooled_mask = masked_adaptive_pool(latents, mask, max_tokens=1)
        self.assertTrue(pooled_mask.item())
        self.assertTrue(torch.allclose(pooled, torch.tensor([[[3.0]]])))

    def test_build_condition_prefixes_role_seed_peer_and_text(self):
        """Both streams receive distinct role tokens and a combined mask."""
        conditioner = DualStreamConditioner(latent_dim=2, condition_dim=4, max_tokens=2)
        base = torch.zeros(1, 3, 4)
        base_mask = torch.ones(1, 3)
        seed = torch.ones(1, 4, 2)
        seed_mask = torch.tensor([[1, 1, 1, 0]])
        peer = torch.ones(1, 5, 2)
        peer_mask = torch.ones(1, 5)

        motif, motif_mask = conditioner.build_condition(
            base, base_mask, seed, seed_mask, peer, peer_mask, MOTIF_ROLE
        )
        vocal, vocal_mask = conditioner.build_condition(
            base, base_mask, seed, seed_mask, peer, peer_mask, VOCAL_ROLE
        )
        self.assertEqual(motif.shape, (1, 8, 4))
        self.assertTrue(torch.equal(motif_mask, vocal_mask))
        self.assertFalse(torch.equal(motif[:, :1], vocal[:, :1]))


if __name__ == "__main__":
    unittest.main()
