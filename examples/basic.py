"""Minimal V4 operation; no research data or accelerator required."""
import torch
from mhd_framework.core import MHD_Node

node = MHD_Node(0, "input", MHD_Node.Message(torch.tensor(2.0)))
assert node.feature_message.current_state.item() == 2.0
print(node.name, node.feature_message.current_state.item())
