"""Graph export adapters.

NetworkX is confined to this boundary and imported lazily; importing this
package pulls in no optional dependency.
"""

from __future__ import annotations

from egrag.adapters.graph.networkx_export import to_graphml

__all__ = ["to_graphml"]
