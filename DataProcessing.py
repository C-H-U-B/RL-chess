import os
import chess.pgn
import chess.engine
import numpy as np
from state import State
from stockfish import Stockfish


def extract_stockfish_data():
    stockfish_engine = Stockfish(path="./stockfish/stockfish-windows-x86-64-avx2")
    input_samples, output_labels = [], []
    game_counter = 0

    for filename in os.listdir('./data'):
        if os.path.isdir(os.path.join('./data', filename)):
            continue

        with open(os.path.join('./data', filename), encoding='ISO-8859-1') as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                if game_counter < 6225:
                    game_counter += 1
                    continue

                board = game.board()
                current_samples = len(input_samples)
                print(f'Parsing game {game_counter}, collected {current_samples} samples')
                for move in game.mainline_moves():
                    board.push(move)
                    stockfish_engine.set_fen_position(board.fen())
                    evaluation = stockfish_engine.get_evaluation()
                    if evaluation['type'] != 'cp':
                        continue

                    serialized_board = State(board).serialize()
                    input_samples.append(serialized_board)
                    output_labels.append(abs(evaluation['value'] / 100))

                game_counter += 1
                if current_samples >= 1_000_000:
                    np.savez('./data/stockfish_processed1M.npz', input_samples, output_labels)
                    return

    return np.array(input_samples), np.array(output_labels)


if __name__ == '__main__':
    extract_stockfish_data()
