"""Style function for the example pipeline in the visualise-the-graph guide."""


def by_stage(*, node, node_class):
    """Dash the outline of nodes tagged as quality control."""
    if node.tags.get("stage") == "qc":
        return {"style": "rounded,dashed,filled"}, node_class, None
    return {}, node_class, None
