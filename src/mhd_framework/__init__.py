"""MHD API V5. The installed package release selects computation semantics."""
from .core import MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph

__version__ = "5.0.0.dev0"
__api_version__ = "V5"
__all__ = ["MHD_Node", "MHD_Edge", "MHD_Topo", "MHD_Graph"]
