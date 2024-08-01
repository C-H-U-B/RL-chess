from minimax import minimaxRoot
import chess, time
from model import Model
import torch
import copy
from state import State
from mcts import GameState, perform_uct_search

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['state_dict'])
    model.eval()

def human_turn(state):
    print(state.board.legal_moves)
    while True:
        try:
            move = input("Enter move: ")
            state.board.push_san(move)
            break
        except Exception:
            print("Invalid move, please try again.")

def mcts_turn(state):
    print("MCTS Turn:")
    computer_move = perform_uct_search(GameState(state=copy.deepcopy(state)), n_simulations=1000)
    if chess.Move.from_uci(str(computer_move) + 'q') in state.board.legal_moves:
        computer_move.promotion = chess.QUEEN
    state.board.push_san(state.board.san(computer_move))
    print("MCTS plays", computer_move)


def minimax_turn(state):
    print("Minimax Turn:")
    move = minimaxRoot(4, state.board, True)
    move = chess.Move.from_uci(str(move))
    state.board.push(move)
    print("Minimax plays", move)

def mcts_versus_human():
    model = Model()
    load_model(model, 'models/mlp-stockfish-1000.pth')
    STATE = State()
    n = 0
    print("Starting game with human:")
    while n < 100:
        start_time = time.time()
        print(STATE.board)
        if n % 2 == 0:
            mcts_turn(STATE)
        else:
            human_turn(STATE)
        iter_duration = time.time() - start_time
        print("turn duration: ", iter_duration)
        n += 1


def mcts_versus_minimax():
    model = Model()
    load_model(model, 'models/mlp-stockfish-new.pth')
    STATE = State()
    n = 0
    print("Starting game between MCTS and Minimax:")
    while n < 100:
        start_time = time.time()
        print(STATE.board)
        if n % 2 == 0:
            mcts_turn(STATE)
        else:
            minimax_turn(STATE)
        n += 1
        iter_duration = time.time() - start_time
        print("turn duration: ", iter_duration)


if __name__ == "__main__":
    mcts_versus_minimax()
