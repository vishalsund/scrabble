"""The hand-played Mini-Scrabble page."""

import json
import threading
import urllib.request

import numpy as np

from scrabble.games.mini_scrabble.gui import Table, make_server, parse_cells, review
from scrabble.games.mini_scrabble.rules import LETTERS, PLAYER, parse_board


def test_opening_placement_needs_two_tiles_through_the_center() -> None:
    board = np.zeros(25, dtype=np.int64)
    one = review(board, np.array([1, 0, 0, 0]), False, [(2, 2, 0)])
    assert one["status"] == "illegal"
    assert one["message"] == "The first placement must use at least two tiles and cover (3,3)."

    pair = review(board, np.array([1, 1, 0, 0]), False, [(2, 2, 0), (2, 3, 1)])
    assert pair == {
        "status": "legal",
        "score": 2,
        "message": "Legal. AB (2) = 2.",
        "words": ["AB"],
    }


def test_explanations_name_the_broken_rule() -> None:
    board = parse_board(
        """
        . . . . .
        . . A B .
        . . C . .
        . . . . .
        . . . . .
        """
    )
    cross = review(board, np.array([0, 0, 0, 1]), True, [(2, 3, LETTERS.index("D"))])
    assert cross["status"] == "legal"
    assert cross["score"] == 4
    assert cross["words"] == ["CD", "BD"]

    diagonal = review(
        _only_center_a(),
        np.array([0, 1, 0, 0]),
        True,
        [(3, 3, LETTERS.index("B"))],
    )
    assert diagonal["message"] == "A new tile must share an edge with a tile already on the board."

    gap = review(
        np.zeros(25, dtype=np.int64),
        np.array([1, 0, 1, 0]),
        False,
        [(2, 0, 0), (2, 2, 2)],
    )
    assert gap["message"] == "An empty square is left between the new tiles."

    aab = review(
        np.zeros(25, dtype=np.int64),
        np.array([2, 1, 0, 0]),
        False,
        [(2, 1, 0), (2, 2, 0), (2, 3, 1)],
    )
    assert aab["message"] == "AAB is not a legal word."


def test_a_side_hook_is_scored_from_the_crossing_word() -> None:
    hook = review(
        _only_center_a(),
        np.array([0, 1, 0, 0]),
        True,
        [(3, 2, LETTERS.index("B"))],
    )
    assert hook["status"] == "legal"
    assert hook["score"] == 2
    assert hook["words"] == ["AB"]


def test_playing_scores_and_an_illegal_click_changes_nothing() -> None:
    table = Table(seed=1)
    game = table.public(ok=True, message="")["game"]
    player = int(game["player"])
    letters = game["racks"][player]
    cells = [
        (2, 2, LETTERS.index(letters[0])),
        (2, 3, LETTERS.index(letters[1])),
    ]
    refused = table.play([(2, 2, LETTERS.index(letters[0]))])
    assert refused["ok"] is False
    assert table.env.score[0, player] == 0
    assert int(table.env.state[0, PLAYER]) == player

    played = table.play(cells)
    assert played["ok"] is True
    assert played["game"]["scores"][player] == 2
    assert played["game"]["player"] == 1 - player
    assert played["game"]["opened"] is True


def test_two_passes_end_as_a_draw_when_nobody_has_scored() -> None:
    table = Table(seed=4)
    table.pass_turn()
    done = table.pass_turn()
    assert done["game"]["over"] is True
    assert done["game"]["hint"] == "Draw 0–0."
    again = table.pass_turn()
    assert again["ok"] is False


def test_the_page_rejects_a_one_tile_opening() -> None:
    table = Table(seed=1)
    server = make_server("127.0.0.1", 0, table)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as response:
            page = response.read().decode()
        assert "<title>Mini-Scrabble</title>" in page
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state") as response:
            payload = json.load(response)
        game = payload["game"]
        letter = game["racks"][game["player"]][0]
        body = json.dumps({"cells": [{"letter": letter, "row": 3, "col": 3}]}).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/play",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            played = json.load(response)
        assert played["ok"] is False
        assert "two tiles" in played["message"]
        assert played["game"]["scores"] == [0, 0]
    finally:
        server.shutdown()
        server.server_close()


def test_parse_cells_uses_one_based_coordinates() -> None:
    cells, error = parse_cells({"cells": [{"letter": "D", "row": 3, "col": 4}]})
    assert error is None
    assert cells == [(2, 3, LETTERS.index("D"))]
    _, error = parse_cells({"cells": [{"letter": "E", "row": 1, "col": 1}]})
    assert error == "Letters are A, B, C, and D."


def _only_center_a() -> np.ndarray:
    board = np.zeros(25, dtype=np.int64)
    board[12] = LETTERS.index("A") + 1
    return board
