"""Synthetic CPU checks for the named graph contract and input geometry."""
import copy
from dataclasses import dataclass
import io

import pytest
import torch
from torch import nn

from mhd_framework.models.graph import NamedModelGraph
from mhd_framework.models.resnet import FlattenViews, ResNetConfig


@dataclass(frozen=True)
class SyntheticConfig:
    name: str = "synthetic_named_contract"
    input_width: int = 4
    feature_width: int = 3
    num_classes: int = 2


def fixture_graph(config=None):
    config = config or SyntheticConfig()
    native = nn.Sequential()
    native.add_module("features", nn.Sequential(
        nn.Linear(config.input_width, config.feature_width), nn.Tanh()))
    native.add_module("logits", nn.Linear(config.feature_width, config.num_classes))
    graph = NamedModelGraph(config, native,
        [("features", native.features), ("logits", native.logits)],
        {"features": config.feature_width}, device="cpu")
    return graph


def step(model, optimizer, inputs):
    optimizer.zero_grad(set_to_none=True)
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    return loss.detach()


def assert_states_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_tiny_adapter_outputs_gradients_and_two_updates():
    torch.manual_seed(541)
    graph = fixture_graph()
    native = copy.deepcopy(graph._native_reference)
    assert sum(p.numel() for p in graph.parameters()) == 23
    assert len(list(graph.parameters())) == len(list(native.parameters()))
    actual_opt = torch.optim.SGD(graph.parameters(), lr=0.03, momentum=0.9)
    native_opt = torch.optim.SGD(native.parameters(), lr=0.03, momentum=0.9)
    for _ in range(2):
        x = torch.randn(2, 4, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        actual_opt.zero_grad(set_to_none=True)
        native_opt.zero_grad(set_to_none=True)
        actual, expected = graph(x), native(ref_x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(x.grad, ref_x.grad, rtol=0, atol=0)
        for (key, p), (ref_key, q) in zip(
                graph._native_reference.named_parameters(), native.named_parameters()):
            assert key == ref_key
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        actual_opt.step()
        native_opt.step()
        assert_states_equal(graph.native_state_dict(), native.state_dict())


def test_named_cut_preserves_output_gradient_and_has_no_stale_reuse():
    torch.manual_seed(542)
    graph = fixture_graph()
    x = torch.randn(2, 4, requires_grad=True)
    full = graph(x)
    expected_grad = torch.autograd.grad(full.sum(), x)[0]
    features = graph.forward_until("features", x)
    actual = graph.forward_from("features", features)
    torch.testing.assert_close(actual, full, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0],
                               expected_grad, rtol=0, atol=0)
    second = x.detach() + 1
    expected_second = graph._native_reference(second)
    actual_second = graph.forward_from("features", graph.forward_until("features", second))
    torch.testing.assert_close(actual_second, expected_second, rtol=0, atol=0)
    assert set(graph.forward(second, return_features=True)) == {"input", "features", "logits"}
    descriptions = graph.describe_nodes()
    assert {entry["id"] for entry in descriptions} == set(graph.endpoint_nodes.values())
    assert all(entry["shape_is_runtime"] for entry in descriptions)


def test_synthetic_checkpoint_resumes_model_optimizer_and_rng_exactly():
    torch.manual_seed(543)
    graph = fixture_graph()
    optimizer = torch.optim.SGD(graph.parameters(), lr=0.03, momentum=0.9)
    step(graph, optimizer, torch.randn(2, 4))
    checkpoint = io.BytesIO()
    torch.save({"schema": "synthetic_contract_fixture_v1",
                "configuration": graph.configuration(),
                "model": graph.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state()}, checkpoint)
    next_inputs = torch.randn(2, 4)
    expected_loss = step(graph, optimizer, next_inputs)
    expected_state = copy.deepcopy(graph.native_state_dict())
    checkpoint.seek(0)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["schema"] == "synthetic_contract_fixture_v1"
    restored = fixture_graph(SyntheticConfig(**saved["configuration"]))
    restored.load_state_dict(saved["model"], strict=True)
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.03, momentum=0.9)
    restored_optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rng"])
    resumed_inputs = torch.randn(2, 4)
    torch.testing.assert_close(resumed_inputs, next_inputs, rtol=0, atol=0)
    torch.testing.assert_close(step(restored, restored_optimizer, resumed_inputs),
                               expected_loss, rtol=0, atol=0)
    assert_states_equal(restored.native_state_dict(), expected_state)
    assert restored.configuration() == graph.configuration()


def test_native_state_roundtrip_is_strict():
    torch.manual_seed(544)
    graph = fixture_graph()
    restored = fixture_graph(SyntheticConfig(**graph.configuration()))
    restored.load_native_state_dict(copy.deepcopy(graph.native_state_dict()))
    x = torch.randn(2, 4)
    torch.testing.assert_close(graph(x), restored(x), rtol=0, atol=0)
    incomplete = copy.deepcopy(graph.native_state_dict())
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(RuntimeError):
        restored.load_native_state_dict(incomplete)


def test_invalid_endpoints_and_duplicate_names_fail_closed():
    graph = fixture_graph()
    for endpoint in ("absent", "logits"):
        with pytest.raises(ValueError):
            graph.forward_from(endpoint, torch.zeros(2, 3))
    for endpoint in ("absent", "input"):
        with pytest.raises(ValueError):
            graph.forward_until(endpoint, torch.zeros(2, 4))
    with pytest.raises(ValueError, match="unique"):
        NamedModelGraph(SyntheticConfig(), nn.Identity(),
            [("features", nn.Identity()), ("features", nn.Identity())], {}, device="cpu")


def test_volume_view_geometry_is_explicit_without_building_heavy_encoder():
    config = ResNetConfig(name="resnet18", spatial_dims=3, in_channels=1,
                          implementation="inflated_3d", views=2)
    flatten = FlattenViews(config)
    volume = torch.arange(2 * 2 * 1 * 3 * 4 * 5).reshape(2, 2, 1, 3, 4, 5)
    torch.testing.assert_close(flatten(volume), volume.flatten(0, 1), rtol=0, atol=0)
    for invalid in (torch.zeros(2, 1, 3, 4, 5),
                    torch.zeros(2, 3, 1, 3, 4, 5),
                    torch.zeros(2, 2, 3, 3, 4, 5)):
        with pytest.raises(ValueError):
            flatten(invalid)
