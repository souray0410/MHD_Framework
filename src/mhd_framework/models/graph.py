"""Named MHD task graphs shared by independent model architectures."""
from dataclasses import asdict
import torch
from mhd_framework.core import MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph


class NamedModelGraph(MHD_Graph):
    def __init__(self, config, reference, operations, channels, *, device='cpu'):
        names = ['input'] + [name for name, _ in operations]
        if len(set(names)) != len(names):
            raise ValueError('Model endpoint names must be unique')
        nodes = [MHD_Node(i, name, MHD_Node.Message(torch.zeros(1)), aggregation='replace')
                 for i, name in enumerate(names)]
        edges = [MHD_Edge(i, name, [MHD_Edge.Operation(module)])
                 for i, (name, module) in enumerate(operations)]
        roles, sorts = [], []
        for i in range(len(edges)):
            role = torch.zeros(len(edges),len(nodes),dtype=torch.long)
            sort = torch.zeros_like(role)
            role[i,i], role[i,i+1], sort[i,i+1] = -1, 1, 1
            roles.append(role); sorts.append(sort)
        topo = MHD_Topo(roles + [-r for r in reversed(roles)], sorts + list(reversed(sorts)))
        super().__init__(set(nodes),set(edges),{topo},device=torch.device(device))
        self.config = config
        self.endpoint_nodes = dict(zip(names,range(len(names))))
        self.feature_channels = channels
        self._definitions = [(i,(i,),i+1) for i in range(len(edges))]
        self._producer_levels = {i+1:i for i in range(len(edges))}
        self.forward_levels = tuple(range(len(edges)))
        self._executed_nodes = set()
        object.__setattr__(self,'_native_reference',reference)

    def describe_nodes(self):
        """Enumerate every real graph node; aliases do not hide internal nodes."""
        return [{'id': node.id, 'name': node.name,
                 'aliases': [k for k,v in self.endpoint_nodes.items() if v == node.id],
                 'shape': list(node.feature_message.current_state.shape),
                 'shape_is_runtime': node.id in self._executed_nodes}
                for node in self._nodes_in_id_order]

    def forward_from(self, endpoint, value, *, return_features=False):
        if endpoint not in self.endpoint_nodes or endpoint == 'logits':
            raise ValueError('Unknown or terminal feature endpoint')
        index = self.endpoint_nodes[endpoint]
        node = self.get_node_by_id(index)
        node.feature_message.current_state = value
        start = self._producer_levels.get(index, -1) + 1
        # A main/skip branch is not a sufficient cut: continuation would require
        # additional tensors. Never silently reuse stale values from a prior call.
        available = {index}
        for _, inputs, output in self._definitions[start:]:
            if not set(inputs).issubset(available):
                raise ValueError('Endpoint is not a complete graph cut; supply the other branch through an explicit project topology')
            available.add(output)
        MHD_Graph.forward(self, levels=self.forward_levels[start:])
        self._executed_nodes = available
        if return_features:
            return {name: self.get_node_by_id(i).feature_message.current_state
                    for name, i in self.endpoint_nodes.items() if i >= index}
        return self.get_node_by_id(self.endpoint_nodes['logits']).feature_message.current_state

    def forward_until(self, endpoint, x):
        """Execute through a named endpoint exactly once, for external graph composition."""
        if endpoint not in self.endpoint_nodes or endpoint == "input":
            raise ValueError("Unknown or initial endpoint")
        self.get_node_by_name("input").feature_message.current_state = x
        stop = self._producer_levels[self.endpoint_nodes[endpoint]] + 1
        MHD_Graph.forward(self, levels=self.forward_levels[:stop])
        self._executed_nodes = {0} | {out for _,_,out in self._definitions[:stop]}
        return self.get_node_by_id(self.endpoint_nodes[endpoint]).feature_message.current_state

    def forward(self, x, *, return_features=False):
        return self.forward_from('input', x, return_features=return_features)

    def native_state_dict(self):
        return self._native_reference.state_dict()

    def load_native_state_dict(self, state):
        return self._native_reference.load_state_dict(state, strict=True)

    def configuration(self):
        return asdict(self.config)
