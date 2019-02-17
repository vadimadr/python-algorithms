from enum import Enum

from algorithms.structures.tree.binary_search_tree import BinarySearchTree


class RBColor(Enum):
    RED = 0
    BLACK = 1


RED = RBColor.RED
BLACK = RBColor.BLACK


def color(node):
    """Utility function to handle NIL nodes"""
    if node is None:
        return BLACK
    return node.color


def set_color(node, new_color):
    """Utility function to handle NIL nodes"""
    if node is None:
        return
    node.color = new_color


class RedBlackTree(BinarySearchTree):
    """
    properties:
    1. Every node is either RED or BLACK
    2. The root is always black
    3. Every leaf (virtual leaf) is black
    4. If node is RED then both children are BLACK
    5. For each node x , all paths to a leve contains same number of black nodes = bh(x)
    """

    def __init__(self, key, *args, **kwargs):
        super().__init__(key, *args, **kwargs)
        # init NIL-like node
        self.color = BLACK
        self._parent = self  # make it parent to itself

    def _swap_nodes(self, other):
        super()._swap_nodes(other)
        self.color, other.color = other.color, self.color

    def add(self, key):
        """Inserts a new node and fixes violations of RB-Tree properties"""
        new_node = super().add(key)
        new_node.color = RED  # inserted node is always RED
        # fix RBT properties after insertion
        z = new_node
        while color(z.parent) is RED:
            # root is black => grand parent exists
            grand_parent = z.parent.parent
            # at most 1 property may be violated (2 or 4) at any iteration.
            parent_is_left = z.parent is grand_parent.left
            if parent_is_left:
                uncle = grand_parent.right
            else:
                uncle = grand_parent.left

            # case 1. Uncle is RED
            if color(uncle) is RED:
                set_color(grand_parent, RED)
                set_color(z.parent, BLACK)
                set_color(uncle, BLACK)
                z = grand_parent
                continue

            # case 2. Uncle is BLACK and z is not the same children as its parent.
            if parent_is_left != (z is z.parent.left):
                z = z.parent.rotate(left=parent_is_left)

            # case 3. Uncle is BLACK and z is at the same side from its parent
            set_color(z.parent, BLACK)
            set_color(grand_parent, RED)
            grand_parent.rotate(left=not parent_is_left)

        self.root.color = BLACK
        return new_node
