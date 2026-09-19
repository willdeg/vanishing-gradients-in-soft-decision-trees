from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Node:
    """
    Node for a soft decision tree.

    Each node starts as a leaf. If splitting it helps validation loss, then it gets two children and uses w and b as gating parameters
    """

    depth: int
    input_dim: int
    is_leaf: bool = True
    z: float = field(init=False)
    w: np.ndarray = field(init=False)
    b: float = field(init=False)
    left = None
    right = None
    parent = None
    is_left_child = None

    def __post_init__(self):
        # Small random values init
        self.z = float(np.random.uniform(-0.01, 0.01))
        self.w = np.random.uniform(-0.01, 0.01, self.input_dim)
        self.b = float(np.random.uniform(-0.01, 0.01))


def sigmoid(z):
    # clip input so exp does not blow up
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(sigmoid_value):
    return sigmoid_value * (1.0 - sigmoid_value)


def forward(node, x):
    """
    Recursively compute the output of the subtree rooted at node
    """

    if node.is_leaf:
        return node.z # irsoy (2012)

    gate = sigmoid(node.w @ x + node.b)
    left_value = forward(node.left, x)
    right_value = forward(node.right, x)
    # the output is a weighted average of the left and right child outputs, where the weights are given by the gate
    return float(gate * left_value + (1.0 - gate) * right_value)


def path_weight(node, x):
    """
    Compute the probability of reaching this node from the root.
    """

    weight = 1.0
    current = node

    while current.parent is not None:
        parent = current.parent
        gate = sigmoid(parent.w @ x + parent.b)
        if current.is_left_child:
            weight *= gate
        else:
            weight *= 1.0 - gate
        current = parent

    return float(weight)


def root_delta(root_output, target, task): # for now we only support classification, but can add regression by changing loss fn
    if task == "classification":
        # With BCE loss and sigmoid output, the derivative is p - y.
        return float(sigmoid(root_output) - target)
    return float(root_output - target)


def prediction_loss(root, x, y, task="classification"):
    scores = np.array([forward(root, row) for row in x], dtype=np.float64)

    if task == "classification":
        probabilities = np.clip(sigmoid(scores), 1e-12, 1.0 - 1e-12)
        return float(-np.mean(y * np.log(probabilities) + (1.0 - y) * np.log(1.0 - probabilities)))

    return float(np.mean((scores - y) ** 2))


def predict(root, x, threshold=0.5):
    scores = np.array([forward(root, row) for row in x], dtype=np.float64)
    probabilities = sigmoid(scores)
    labels = (probabilities >= threshold).astype(np.int64)
    return labels, probabilities


def initialize_root(input_dim, positive_rate):
    """
    Initialize the root logit using the training-set positive class rate.
    """

    positive_rate = float(np.clip(positive_rate, 1e-6, 1.0 - 1e-6))
    root = Node(depth=0, input_dim=input_dim)
    root.z = float(np.log(positive_rate / (1.0 - positive_rate)))
    return root


def update_node(
    node,
    x,
    target,
    root,
    learning_rate,
    use_depth_scaling,
    grad_trace=None,
):
    """
    Update one candidate split using one sample.

    This is where the ordinary and scaled trees differ. The ordinary tree uses
    scale = 1, while the scaled tree uses scale = 2^depth.
    """

    root_output = forward(root, x)
    delta_root = root_delta(root_output, target, task="classification")
    delta_node = delta_root * path_weight(node, x)

    gate = sigmoid(node.w @ x + node.b)
    left_output = forward(node.left, x)
    right_output = forward(node.right, x)
    beta = delta_node * (left_output - right_output)

    grad_w = beta * sigmoid_derivative(gate) * x
    grad_b = beta * sigmoid_derivative(gate)
    grad_z_left = delta_node * gate
    grad_z_right = delta_node * (1.0 - gate)

    scale = float(2**node.depth) if use_depth_scaling else 1.0 # scaling factor, can be turned on or off

    if grad_trace is not None:
        grad_trace.append(
            {
                "w": float(np.linalg.norm(scale * grad_w)),
                "b": float(abs(scale * grad_b)),
                "z_left": float(abs(scale * grad_z_left)),
                "z_right": float(abs(scale * grad_z_right)),
            }
        )

    node.w -= learning_rate * scale * grad_w
    node.b -= learning_rate * scale * grad_b
    node.left.z -= learning_rate * scale * grad_z_left
    node.right.z -= learning_rate * scale * grad_z_right


def train_node(node,
    x,
    y,
    root,
    learning_rate,
    max_iter,
    use_depth_scaling,
    track_grads=False,):
    grad_trace = []

    for _ in range(max_iter):
        # shuffle each epoch so we do not always see the samples in the same order
        order = np.random.permutation(len(x))
        for idx in order:
            update_node(
                node=node,
                x=x[idx],
                target=y[idx],
                root=root,
                learning_rate=learning_rate,
                use_depth_scaling=use_depth_scaling,
                grad_trace=grad_trace if track_grads else None,
            )

    return grad_trace


def snapshot_node(node):
    """
    Save a node's current state so we can restore it if a split is rejected.
    """

    return {
        "is_leaf": node.is_leaf,
        "z": node.z,
        "w": node.w.copy(),
        "b": node.b,
        "left": node.left,
        "right": node.right,
        "parent": node.parent,
        "is_left_child": node.is_left_child,
    }


def restore_node(node, snapshot):
    node.is_leaf = snapshot["is_leaf"]
    node.z = snapshot["z"]
    node.w = snapshot["w"].copy()
    node.b = snapshot["b"]
    node.left = snapshot["left"]

    node.right = snapshot["right"]
    node.parent = snapshot["parent"]

    node.is_left_child = snapshot["is_left_child"]


def learn_soft_tree(
    node,
    x_train,
    y_train,
    x_val,
    y_val,
    root,
    learning_rate=0.01,
    max_iter=25,
    max_depth=10,
    use_depth_scaling=False,
    grad_recorder=None,
    verbose=False,

):
    """
    Greedy tree growth:
    try a split, train it, keep it only if validation loss improves.
    """

    if node.depth >= max_depth:
        return {} if grad_recorder is None else grad_recorder

    if grad_recorder is None:
        grad_recorder = {}

    loss_before = prediction_loss(root, x_val, y_val, task="classification")
    snapshot = snapshot_node(node)

    # Turn the current leaf into a candidate split.
    node.is_leaf = False
    node.left = Node(node.depth + 1, x_train.shape[1])
    node.right = Node(node.depth + 1, x_train.shape[1])
    node.left.parent = node
    node.right.parent = node
    node.left.is_left_child = True
    node.right.is_left_child = False

    # Start the child leaves near the parent value
    node.left.z = node.z + float(np.random.uniform(-0.01, 0.01))
    node.right.z = node.z + float(np.random.uniform(-0.01, 0.01))

    node_trace = train_node(
        node=node,
        x=x_train,
        y=y_train,
        root=root,
        learning_rate=learning_rate,
        max_iter=max_iter,
        use_depth_scaling=use_depth_scaling,
        track_grads=True,
    )
    grad_recorder.setdefault(node.depth, []).append(node_trace)

    loss_after = prediction_loss(root, x_val, y_val, task="classification")

    if loss_after < loss_before: # note we want strictly better
        if verbose:
            print(f"Accepted split at depth {node.depth}")
        learn_soft_tree(node.left, x_train,
                        y_train, x_val, y_val, root, learning_rate, max_iter, max_depth, use_depth_scaling, grad_recorder, verbose)
        learn_soft_tree(node.right, x_train, y_train, x_val, y_val, root, learning_rate, max_iter, max_depth, use_depth_scaling, grad_recorder, verbose)
    else:
        if verbose:
            print(f"Rejected split at depth {node.depth}")
        restore_node(node, snapshot)

    return grad_recorder


def plot_gradient_boxplot(
    grad_recorder,
    key="w",
    use_log=True,

    save_path=None,
):
    depths = []
    values = []

    for depth in sorted(grad_recorder):
        traces = grad_recorder[depth]
        per_node_means = []
        for trace in traces:
            vals = [step[key] for step in trace if key in step]
            if vals:
                per_node_means.append(float(np.mean(vals)))
        if per_node_means:
            depths.append(depth)
            values.append(per_node_means)


    if not values:
        return

    # plot boxplot of the average gradient norms per node at each depth
    plt.figure(figsize=(8, 5))
    plt.boxplot(values, tick_labels=[f"depth {depth}" for depth in depths])

    if use_log:
        plt.yscale("log")

    plt.xlabel("Depth")
    plt.ylabel(f"Average gradient norm per node ({key})")
    plt.title("Gradient Norm Distribution by Depth")
    plt.grid(alpha=0.3)

    if save_path is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
