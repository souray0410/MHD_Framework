"""A two-operation hypergraph using synthetic CPU tensors."""
import torch
from torch import nn
from mhd_framework.core import MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph

linear = nn.Linear(1, 1, bias=False)
nodes = {
    MHD_Node(0, "input", MHD_Node.Message(torch.zeros(2, 1))),
    MHD_Node(1, "prediction", MHD_Node.Message(torch.zeros(2, 1))),
    MHD_Node(2, "loss", MHD_Node.Message(torch.zeros(()))),
}
edges = {
    MHD_Edge(0, "linear", [MHD_Edge.Operation(linear)]),
    MHD_Edge(1, "loss", [MHD_Edge.Operation(lambda x: x.square().mean())]),
}
forward_0 = torch.tensor([[-1, 1, 0], [0, 0, 0]])
forward_1 = torch.tensor([[0, 0, 0], [0, -1, 1]])
sort_0 = torch.tensor([[0, 1, 0], [0, 0, 0]])
sort_1 = torch.tensor([[0, 0, 0], [0, 0, 1]])
topology = MHD_Topo(
    [forward_0, forward_1, -forward_1, -forward_0],
    [sort_0, sort_1, sort_1, sort_0],
)
graph = MHD_Graph(nodes, edges, {topology}, device="cpu")
graph.get_node_by_name("input").feature_message.current_state = torch.ones(2, 1)
graph.forward(levels=[0, 1])
graph.backward(levels=[2, 3])
assert linear.weight.grad is not None
print(graph.get_node_by_name("loss").feature_message.current_state)
