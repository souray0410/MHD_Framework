"""Public package surface and synthetic core gradient checks."""
import pytest
import torch

from mhd_framework import MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph
from mhd_framework.models import create_model


def chain_graph():
    nodes = {
        MHD_Node(i, f"n{i}", MHD_Node.Message(torch.tensor(0.0)))
        for i in range(3)
    }
    edges = {
        MHD_Edge(0, "double", [MHD_Edge.Operation(lambda x: 2 * x)]),
        MHD_Edge(1, "square", [MHD_Edge.Operation(lambda x: x.square())]),
    }
    first = torch.tensor([[-1, 1, 0], [0, 0, 0]])
    second = torch.tensor([[0, 0, 0], [0, -1, 1]])
    first_order = torch.tensor([[0, 1, 0], [0, 0, 0]])
    second_order = torch.tensor([[0, 0, 0], [0, 0, 1]])
    topology = MHD_Topo(
        [first, second, -second, -first],
        [first_order, second_order, second_order, first_order],
    )
    return MHD_Graph(nodes, edges, {topology}, device="cpu")


def test_scalar_core_backward_matches_native_expression():
    graph = chain_graph()
    value = torch.tensor(2.0, requires_grad=True)
    graph.get_node_by_id(0).feature_message.current_state = value
    graph.forward([0, 1]).backward([2, 3])
    torch.testing.assert_close(graph.get_node_by_id(2).feature_message.current_state,
                               (2 * value).square())
    torch.testing.assert_close(value.grad, 8 * value.detach())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_explicit_vector_cotangent_matches_native_autograd(dtype):
    graph = chain_graph()
    value = torch.tensor([1.0, 2.0], dtype=dtype, requires_grad=True)
    cotangent = torch.tensor([1.0, -2.0], dtype=dtype)
    graph.get_node_by_id(0).feature_message.current_state = value
    graph.get_node_by_id(2).gradient_message.update_initial(cotangent)
    graph.forward([0, 1]).backward([2, 3])
    native = value.detach().clone().requires_grad_()
    expected, = torch.autograd.grad((2 * native).square(), native, cotangent)
    torch.testing.assert_close(value.grad, expected)


def test_factory_rejects_unexported_families_before_construction():
    with pytest.raises(ValueError, match="ResNet"):
        create_model({"name": "unsupported_family"})
    with pytest.raises(TypeError, match="dictionary"):
        create_model(None)
