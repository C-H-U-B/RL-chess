from model import Model
from torch import Tensor
import torch
import collections
import copy
import random
import numpy as np

# Load the pre-trained model
model = Model()
model.load_state_dict(torch.load('models/mlp-stockfish-new.pth', map_location=torch.device('cpu'))['state_dict'])
model.eval()

current_child_idx = 0


class TreeNode:
    def __init__(self, state, move, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.is_expanded = False
        self.children = {}
        self.prior_probs = np.full(self.state.children_len, 2, dtype=np.float32)
        self.total_values = np.zeros(self.state.children_len, dtype=np.float32)
        self.visit_counts = np.zeros(self.state.children_len, dtype=np.float32)
        self.has_won = False

    @property
    def visit_count(self):
        return self.parent.visit_counts[self.move]

    @visit_count.setter
    def visit_count(self, value):
        self.parent.visit_counts[self.move] = value

    @property
    def total_value(self):
        return self.parent.total_values[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.total_values[self.move] = value

    def compute_q_values(self):  # Compute the average evaluation values
        return self.total_values / (1 + self.visit_counts)

    def compute_u_values(self):  # Compute the exploration bonuses
        return self.prior_probs * np.sqrt(self.visit_count / (1.0 + self.visit_counts))

    def select_best_child(self):
        return np.argmax(self.compute_q_values() + self.compute_u_values())

    def find_leaf_node(self):
        current_node = self
        if current_node.parent.parent is None:
            current_node.is_expanded = True
        while current_node.is_expanded:
            try:
                global current_child_idx
                best_move = current_node.select_best_child()
                current_child_idx = best_move
                current_node = current_node.add_child_if_needed(best_move)
            except:
                continue
        return current_node

    def expand_node(self, prior_probs):
        self.is_expanded = True
        state_copy = copy.deepcopy(self.state.state)
        while not state_copy.board.is_game_over():
            move = state_copy.board.san(select_random_move(state_copy))
            state_copy.board.push_san(move)
        result = state_copy.board.result()
        self.has_won = (result == '1-0' and state_copy.board.turn) or \
                       (result == '0-1' and not state_copy.board.turn) or \
                       result == '1/2-1/2'
        if self.has_won:
            self.total_value += 1
        self.visit_count += 1

    def add_child_if_needed(self, move):
        if move not in self.children or self.children[move] is None:
            self.children[move] = TreeNode(self.state.play(move), move, parent=self)
        return self.children[move]

    def back_propagate(self, value_estimate: float):
        current_node = self
        while current_node.parent.parent is not None:
            current_node.parent.total_value = np.sum(current_node.parent.total_values, dtype=np.float32)
            if current_node.parent.has_won:
                current_node.parent.total_value += 1
            current_node.parent.visit_count = np.sum(current_node.parent.visit_counts, dtype=np.float32) + 1
            current_node = current_node.parent


def select_random_move(state):
    return random.choice([move for move in state.legal_moves])


def evaluate_move_with_network(state):
    successors = []
    for move in state.legal_moves:
        state.board.push_san(str(move))
        successors.append(torch.argmax(model(Tensor(state.serialize()))))
        state.board.pop()

    if 0 in successors:
        return state.legal_moves[successors.index(0)]
    elif 2 in successors:
        return state.legal_moves[successors.index(2)]
    elif successors:
        return successors[0]
    else:
        return []


class RootNode:
    def __init__(self):
        self.parent = None
        self.total_values = collections.defaultdict(float)
        self.visit_counts = collections.defaultdict(float)


class NeuralNetwork:
    @classmethod
    def evaluate(cls, state):
        return 2.0, 1.0


def expand_children(root, num_children):
    for i in range(num_children):
        root.children[i] = TreeNode(root.state.play(i), i, parent=root)
        root.children[i].is_expanded = True
        state_copy = copy.deepcopy(root.children[i].state.state)
        while not state_copy.board.is_game_over():
            move = state_copy.board.san(select_random_move(state_copy))
            state_copy.board.push_san(move)
        result = state_copy.board.result()
        root.children[i].has_won = (result == '1-0' and state_copy.board.turn) or \
                                   (result == '0-1' and not state_copy.board.turn) or \
                                   result == '1/2-1/2'
        if root.children[i].has_won:
            root.children[i].total_value += 1
        root.children[i].visit_count += 1
        root.children[i].back_propagate(1)


def perform_uct_search(state, n_simulations):
    root = TreeNode(state, move=None, parent=RootNode())
    root.is_expanded = True
    expand_children(root, root.state.children_len)

    for _ in range(n_simulations):
        leaf = root.find_leaf_node()
        prior_probs, value_estimate = NeuralNetwork.evaluate(leaf.state)
        leaf.expand_node(prior_probs)
        leaf.back_propagate(value_estimate)
    return state.state.legal_moves[np.argmax(root.total_values)]


class GameState:
    def __init__(self, turn=1, state=None):
        self.turn = turn
        self.state = state
        self.children_len = len(state.legal_moves) if state.legal_moves else 0

    def play(self, move):
        state_copy = copy.deepcopy(self.state)
        move_str = self.state.board.san(self.state.legal_moves[move])
        state_copy.board.push_san(move_str)
        return GameState(-self.turn, state_copy)
