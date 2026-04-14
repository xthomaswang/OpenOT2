"""OpenOT2 Protocol Designer — canonical IR, validation, compilation, and visualization."""

from openot2.designer.ir import (
    DeckSetup,
    Edge,
    LabwareEntry,
    LiquidEntry,
    ModuleEntry,
    Node,
    NodeKind,
    PipetteEntry,
    ProtocolIR,
)
from openot2.designer.compiler import compile_graph, compile_linear
from openot2.designer.validator import validate
from openot2.designer.visualize import graph, summarize_node, timeline, to_mermaid
from openot2.designer.extensions import (
    build_capture_predict_branch,
    capture_artifact_key,
    get_dataflow,
    link_capture_to_predict,
    link_predict_to_branch,
    make_branch,
    make_capture,
    make_predict,
    predict_result_key,
)

__all__ = [
    # IR
    "DeckSetup",
    "Edge",
    "LabwareEntry",
    "LiquidEntry",
    "ModuleEntry",
    "Node",
    "NodeKind",
    "PipetteEntry",
    "ProtocolIR",
    # Compiler
    "compile_graph",
    "compile_linear",
    # Validator
    "validate",
    # Visualization
    "graph",
    "summarize_node",
    "timeline",
    "to_mermaid",
    # Extensions
    "build_capture_predict_branch",
    "capture_artifact_key",
    "get_dataflow",
    "link_capture_to_predict",
    "link_predict_to_branch",
    "make_branch",
    "make_capture",
    "make_predict",
    "predict_result_key",
]
