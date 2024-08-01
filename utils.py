import base64
import chess.svg

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"

def to_svg(state):
    return base64.b64encode(chess.svg.board(board=state.board).encode('utf-8')).decode('utf-8')


def stockfish_treshold(x):
    """
    if x > 1.5 -> Winning Label
    if x < -1.5 -> Losing Label
    if -1.5 <= x <= 1.5 -> Draw Label
    """

    if x > 1.5:
        return 0
    elif x < -1.5:
        return 1
    else:
        return 2