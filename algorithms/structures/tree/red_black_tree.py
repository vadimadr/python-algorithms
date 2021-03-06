from enum import Enum

from .binary_search_tree import BinarySearchTree


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
        """
        Inserts a new node and fixes violations of RB-Tree properties

        Insertion fix-up. Capitals are black nodes, lowercase are red nodes. z is the newly inserted node.
        ">" points to the node that potentially breaks the RB properties. When z is the right child, tree must be
        reflected symmetrically

                case 1                  case 2                       case 3
            G            g         G              G             G            > P
           / \          / \       / \            / \           / \            / \
          p   u   ->   P   U     p   U   ->     p   U         p   U   ->     z   g
          |            |          \            /             /                    \
        > z            z         > z        > z           > z                      U

        """
        new_node = super().add(key)
        new_node.color = RED  # inserted node is always RED
        # fix RBT properties after insertion
        z = new_node
        while color(z.parent) is RED:
            # at most 1 property may be violated (2 or 4) at any iteration.
            # root is black => grand parent exists
            grand_parent = z.parent.parent
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
            # e.g. one is the left child and another is the right or vice versa.
            if parent_is_left != (z is z.parent.left):
                z = z.parent.rotate(left=parent_is_left)

            # case 3. Uncle is BLACK and z is the same children as its parent.
            set_color(z.parent, BLACK)
            set_color(grand_parent, RED)
            grand_parent.rotate(left=not parent_is_left)

        self.root.color = BLACK
        return new_node

    def delete(self, key):
        # standard tree deletion procedure
        v, y, x = super().delete(key)

        if y.color is BLACK:
            # y was black => it may cause path containing y to have
            # one black node less. Need to restore properties
            self._restore_delete(x, y.parent)

    def _restore_delete(self, x, parent=None):
        """
        Deletion fix-up. X is the node that takes place of transplanted (or deleted) node.
        "tick" symbol means "keep the same color". Reflect tree symmetrically if x is the right child.

                  case 1                     case 2
            P                B         p'           > P
           / \              / \       / \            / \
          X   b     ->     p   V     X   B     ->   X   b
             / \          / \           / \            / \
            U   V        X   U         U   V          U   V


                 case 3                       case 4
            p'           > P'          p'               b'
           / \            / \         / \              / \
          X   B     ->   X   U       X   B     ->     P   V
             / \              \         / \          / \
            u   V              b       u'  v        X   u'

        """
        # x points to the node that takes place of the y
        # it may be None iff y had no children
        while parent and color(x) is BLACK:
            is_left_child = parent.left is x
            brother = parent.right if is_left_child else parent.left
            # case 1. Brother is red
            if color(brother) is RED:
                set_color(parent, RED)
                set_color(brother, BLACK)
                parent = parent.rotate(is_left_child)
                brother = parent.right if is_left_child else parent.left

            # brother is black
            u, v = brother.left, brother.right
            if not is_left_child:
                u, v = v, u

            # case 2 both children are black
            if color(u) == color(v) == BLACK:
                set_color(brother, RED)
                x, parent = parent, parent.parent
            else:
                # case 3. left children is black
                if color(v) is BLACK:
                    set_color(u, BLACK)
                    set_color(brother, RED)
                    brother.rotate(not is_left_child)
                    brother = parent.right if is_left_child else parent.left
                    u, v = brother.left, brother.right
                    if not is_left_child:
                        u, v = v, u

                # case 4. right children is black
                set_color(brother, color(parent))
                set_color(parent, BLACK)
                set_color(v, BLACK)
                parent.rotate(is_left_child)
                x = self.root
                parent = None

        set_color(x, BLACK)
        set_color(self.root, BLACK)

    def __repr__(self):
        return "RBTree({}, {})".format(self.key, "R" if self.color is RED else "B")
