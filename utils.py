from __future__ import annotations
from typing import Iterable, Protocol, TypeVar, Generic

class TreeNode(Protocol):
    name: str
    children: Iterable[TreeNode]


T = TypeVar("T", bound=TreeNode)

class TreeFormatter(Generic[T]):
    def __init__(self, root: T) -> None:
        self._root = root

    def format(self) -> str:
        return "\n".join(self._lines())

    def __str__(self) -> str:
        return self.format()

    def _lines(self) -> Iterable[str]:
        if not self._root.name:
            yield "."
            return
        yield self._root.name
        yield from self._render_children(self._root, prefix="")

    def _render_children(self, node: TreeNode, prefix: str) -> Iterable[str]:
        children = list(node.children)
        for idx, child in enumerate(children):
            is_last = idx == len(children) - 1
            connector = "└── " if is_last else "├── "
            yield f"{prefix}{connector}{child.name}"
            child_prefix = prefix + ("    " if is_last else "│   ")
            yield from self._render_children(child, child_prefix)