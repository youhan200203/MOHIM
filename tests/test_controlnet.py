"""Tests for the paper-faithful DiT ControlNet topology."""

import copy
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("PyTorch is not installed in this local test environment") from exc
from torch import nn

from mohim.controlnet import (
    AceStepDiTControlNet,
    CONTROLNET_SCHEMA,
    TopKCQTMelodyEncoder,
    align_repeated_topk_cqt,
    make_silence_context,
)
from mohim.controlnet_training import flow_matching_step, load_silence_latent


class _Transpose(nn.Module):
    def forward(self, values):
        return values.transpose(1, 2)


class _TimeEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.projection = nn.Linear(1, hidden_size)

    def forward(self, timestep):
        embedding = self.projection(timestep[:, None])
        return embedding, embedding[:, None].expand(-1, 6, -1)


class _Rotary(nn.Module):
    def forward(self, hidden_states, position_ids):
        return None


class _FakeLayer(nn.Module):
    def __init__(self, hidden_size, attention_type):
        super().__init__()
        self.attention_type = attention_type
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        timestep_projection,
        attention_mask,
        position_ids,
        past_key_values,
        output_attentions,
        use_cache,
        cache_position,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        return (hidden_states + torch.tanh(self.projection(hidden_states)),)


class _FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 8
        self.config = types.SimpleNamespace(
            hidden_size=hidden_size,
            _attn_implementation="flash_attention_2",
            use_sliding_window=False,
            sliding_window=None,
        )
        self.patch_size = 2
        self.proj_in = nn.Sequential(
            _Transpose(), nn.Conv1d(6, hidden_size, 2, stride=2), _Transpose()
        )
        self.time_embed = _TimeEmbedding(hidden_size)
        self.time_embed_r = _TimeEmbedding(hidden_size)
        self.condition_embedder = nn.Linear(5, hidden_size)
        self.rotary_emb = _Rotary()
        self.layers = nn.ModuleList(
            [_FakeLayer(hidden_size, "full_attention") for _ in range(4)]
        )
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 2, hidden_size))
        self.norm_out = nn.LayerNorm(hidden_size)
        self.proj_out = nn.Sequential(
            _Transpose(), nn.ConvTranspose1d(hidden_size, 2, 2, stride=2), _Transpose()
        )

    def forward(
        self,
        *,
        hidden_states,
        timestep,
        timestep_r,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        context_latents,
    ):
        original_length = hidden_states.shape[1]
        hidden_states = self.proj_in(torch.cat([context_latents, hidden_states], dim=-1))
        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)
        temb_t, timestep_projection_t = self.time_embed(timestep)
        temb_r, timestep_projection_r = self.time_embed_r(timestep - timestep_r)
        temb = temb_t + temb_r
        timestep_projection = timestep_projection_t + timestep_projection_r
        position_ids = torch.arange(hidden_states.shape[1]).unsqueeze(0)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                None,
                timestep_projection,
                None,
                position_ids,
                None,
                False,
                False,
                None,
                encoder_hidden_states,
                None,
            )[0]
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale) + shift
        return self.proj_out(hidden_states)[:, :original_length], None


class _FakeFlowModel(nn.Module):
    def forward(self, *, hidden_states, **kwargs):
        self.last_context_latents = kwargs["context_latents"].detach().clone()
        return hidden_states * 0.25, None


class ControlNetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.decoder = _FakeDecoder()
        self.reference = copy.deepcopy(self.decoder)
        self.model = AceStepDiTControlNet(
            self.decoder, copy_blocks=2, pitch_embedding_dim=4, gradient_checkpointing=False
        )
        self.inputs = {
            "hidden_states": torch.randn(1, 8, 2),
            "timestep": torch.tensor([0.7]),
            "timestep_r": torch.tensor([0.7]),
            "attention_mask": torch.ones(1, 8),
            "encoder_hidden_states": torch.randn(1, 3, 5),
            "encoder_attention_mask": torch.ones(1, 3),
            "context_latents": torch.randn(1, 8, 4),
        }
        self.melody = torch.randint(0, 128, (1, 16, 8))

    def test_zero_initialization_preserves_backbone_output(self):
        expected = self.reference(**self.inputs)[0]
        actual = self.model(
            **self.inputs, melody_pitch_indices=self.melody, control_scale=1.0
        )[0]
        torch.testing.assert_close(actual, expected)

    def test_backbone_is_frozen_and_zero_bridge_receives_gradient(self):
        output = self.model(
            **self.inputs, melody_pitch_indices=self.melody, control_scale=1.0
        )[0]
        output.square().mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in self.decoder.parameters()))
        self.assertIsNotNone(self.model.control_blocks[0].after_proj.weight.grad)
        self.assertGreater(self.model.trainable_parameter_count(), 0)

    def test_first_copied_block_receives_same_input_as_first_frozen_block(self):
        captured = {}

        def capture(name):
            def hook(_module, args):
                captured[name] = args[0].detach().clone()
            return hook

        handles = [
            self.decoder.layers[0].register_forward_pre_hook(capture("frozen")),
            self.model.control_blocks[0].copied_block.register_forward_pre_hook(
                capture("copied")
            ),
        ]
        try:
            self.model(
                **self.inputs, melody_pitch_indices=self.melody, control_scale=1.0
            )
        finally:
            for handle in handles:
                handle.remove()

        torch.testing.assert_close(captured["copied"], captured["frozen"])

    def test_copied_residual_is_added_after_matching_frozen_block(self):
        first_control = self.model.control_blocks[0]
        with torch.no_grad():
            first_control.after_proj.weight.copy_(torch.eye(8))
            first_control.after_proj.bias.zero_()

        captured = {}

        def capture_output(name):
            def hook(_module, _args, output):
                captured[name] = output[0].detach().clone()
            return hook

        def capture_input(name):
            def hook(_module, args):
                captured[name] = args[0].detach().clone()
            return hook

        handles = [
            self.decoder.layers[0].register_forward_hook(capture_output("frozen_0")),
            first_control.copied_block.register_forward_hook(capture_output("copied_0")),
            self.decoder.layers[1].register_forward_pre_hook(capture_input("frozen_1_input")),
        ]
        try:
            self.model(
                **self.inputs, melody_pitch_indices=self.melody, control_scale=1.0
            )
        finally:
            for handle in handles:
                handle.remove()

        torch.testing.assert_close(
            captured["frozen_1_input"], captured["frozen_0"] + captured["copied_0"]
        )

    def test_control_checkpoint_excludes_frozen_backbone(self):
        state = self.model.control_state_dict()
        self.assertEqual(state["schema"], CONTROLNET_SCHEMA)
        self.assertEqual(state["copy_blocks"], 2)
        self.assertNotIn("base_decoder", state)
        restored = AceStepDiTControlNet(
            copy.deepcopy(self.reference),
            copy_blocks=2,
            pitch_embedding_dim=4,
            gradient_checkpointing=False,
        )
        restored.load_control_state_dict(state)

    def test_legacy_control_checkpoint_is_rejected(self):
        state = self.model.control_state_dict()
        state["schema"] = "ace_step_dit_controlnet_topk_cqt_v1"
        restored = AceStepDiTControlNet(
            copy.deepcopy(self.reference),
            copy_blocks=2,
            pitch_embedding_dim=4,
            gradient_checkpointing=False,
        )
        with self.assertRaisesRegex(ValueError, "incompatible with the parallel"):
            restored.load_control_state_dict(state)

    def test_melody_encoder_matches_requested_token_length(self):
        encoder = TopKCQTMelodyEncoder(16, pitch_embedding_dim=4)
        output = encoder(torch.randint(0, 128, (2, 31, 8)), target_length=7)
        self.assertEqual(output.shape, (2, 7, 16))

    def test_repeated_cqt_is_phase_aligned_to_motif_anchor(self):
        unaligned = torch.arange(12).remainder(4).unsqueeze(1)
        aligned = align_repeated_topk_cqt(
            unaligned,
            motif_start_sec=1.0,
            motif_end_sec=5.0,
            frame_rate=1.0,
        )
        self.assertEqual(aligned.squeeze(1).tolist(), [3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2])

    def test_silence_context_has_source_and_ones_mask(self):
        silence = torch.randn(1, 3, 64)
        context = make_silence_context(
            silence,
            batch_size=2,
            target_length=7,
            device="cpu",
            dtype=torch.float32,
        )
        self.assertEqual(context.shape, (2, 7, 128))
        self.assertTrue(torch.all(context[..., 64:] == 1))

    def test_load_silence_latent_transposes_channel_first_layout(self):
        channel_first = torch.randn(1, 64, 11)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "silence_latent.pt"
            torch.save(channel_first, path)
            loaded = load_silence_latent(path)
        self.assertEqual(loaded.shape, (1, 11, 64))
        torch.testing.assert_close(loaded, channel_first.transpose(1, 2))

    def test_flow_matching_step_reuses_fixed_noise_and_timestep(self):
        batch = {
            "target_latents": torch.randn(1, 8, 64),
            "context_latents": torch.randn(1, 8, 128),
            "attention_mask": torch.ones(1, 8),
            "encoder_hidden_states": torch.randn(1, 3, 5),
            "encoder_attention_mask": torch.ones(1, 3),
            "melody_pitch_indices": torch.randint(0, 128, (1, 16, 8)),
        }
        noise = torch.randn_like(batch["target_latents"])
        timestep = torch.tensor([0.5])
        model = _FakeFlowModel()
        first = flow_matching_step(
            model,
            batch,
            timestep_mu=-0.4,
            timestep_sigma=1.0,
            noise=noise,
            timestep=timestep,
        )
        torch.manual_seed(1234)
        second = flow_matching_step(
            _FakeFlowModel(),
            batch,
            timestep_mu=-0.4,
            timestep_sigma=1.0,
            noise=noise,
            timestep=timestep,
        )
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        torch.testing.assert_close(model.last_context_latents, batch["context_latents"])

    def test_flow_matching_step_rejects_invalid_fixed_inputs(self):
        batch = {
            "target_latents": torch.randn(1, 8, 64),
            "context_latents": torch.randn(1, 8, 128),
            "attention_mask": torch.ones(1, 8),
            "encoder_hidden_states": torch.randn(1, 3, 5),
            "encoder_attention_mask": torch.ones(1, 3),
            "melody_pitch_indices": torch.randint(0, 128, (1, 16, 8)),
        }
        with self.assertRaisesRegex(ValueError, "Fixed noise shape"):
            flow_matching_step(
                _FakeFlowModel(),
                batch,
                timestep_mu=-0.4,
                timestep_sigma=1.0,
                noise=torch.randn(1, 7, 64),
                timestep=torch.tensor([0.5]),
            )
        with self.assertRaisesRegex(ValueError, "Fixed timestep values"):
            flow_matching_step(
                _FakeFlowModel(),
                batch,
                timestep_mu=-0.4,
                timestep_sigma=1.0,
                noise=torch.randn_like(batch["target_latents"]),
                timestep=torch.tensor([1.5]),
            )


if __name__ == "__main__":
    unittest.main()
