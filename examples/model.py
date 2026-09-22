"""Synthetic CPU model update and strict selected-weight reload; no downloads."""
import copy
import torch
from mhd_framework.models import create_model

torch.manual_seed(19)
torch.set_num_threads(2)
model = create_model(dict(name="resnet18", views=1))
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
inputs = torch.randn(2, 3, 32, 32)
optimizer.zero_grad(set_to_none=True)
model(inputs).square().mean().backward()
optimizer.step()
selected_weights = copy.deepcopy(model.state_dict())
restored = create_model(model.configuration())
restored.load_state_dict(selected_weights, strict=True)
model.eval()
restored.eval()
with torch.no_grad():
    torch.testing.assert_close(model(inputs), restored(inputs), rtol=0, atol=0)
print("V5 model update and strict weight reload passed")
