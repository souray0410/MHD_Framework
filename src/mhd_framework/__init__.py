"""MHD API V4. The installed package release selects computation semantics."""
from .core import MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph

__version__ = "4"
__api_version__ = "V4"
__all__ = ["MHD_Node", "MHD_Edge", "MHD_Topo", "MHD_Graph"]
