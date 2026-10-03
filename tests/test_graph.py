"""Import check for the graph assembly placeholder."""

from customer_support import graph


def test_graph_module_is_importable() -> None:
    assert graph is not None
