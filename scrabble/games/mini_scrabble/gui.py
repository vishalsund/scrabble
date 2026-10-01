"""A local page for playing Mini-Scrabble by hand.

Run ``python -m scrabble.games.mini_scrabble.gui``. Both players sit at the
same board. The page asks this process which placements are legal; the
environment is what accepts a move.
"""

from __future__ import annotations

import argparse
import errno
import json
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

from scrabble.games.mini_scrabble.env import MiniScrabbleEnv
from scrabble.games.mini_scrabble.rules import (
    AXIS,
    BAG,
    BOARD_SIZE,
    CENTER,
    LETTERS,
    LETTERS_AT,
    N_LETTERS,
    N_SQUARES,
    N_TILES,
    OPENED,
    PASS,
    PASSES,
    PLAYER,
    RACK,
    SQUARES,
    _run,
    action_for,
    action_name,
    is_legal_word,
    placement_score,
)

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mini-Scrabble</title>
<style>
  :root {
    color-scheme: light;
    --ink: #2b2622;
    --paper: #f4efe6;
    --board: #e7dcc8;
    --line: #cbbfa8;
    --muted: #6f675e;
    --ok: #1f7a4d;
    --bad: #9c2f2f;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font: 16px/1.4 "Iowan Old Style", Palatino, Georgia, serif;
    color: var(--ink);
    background: var(--paper);
  }
  main {
    max-width: 760px;
    margin: 0 auto;
    padding: 20px 16px 40px;
  }
  header {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: end;
    margin-bottom: 16px;
  }
  h1 { font-size: 28px; margin: 0; font-weight: 650; }
  form { display: flex; gap: 8px; }
  input, button {
    font: inherit;
    border-radius: 8px;
    border: 1px solid var(--line);
  }
  button:disabled { cursor: default; }
  .actions button:disabled { opacity: 0.45; }
  .player:not(.turn) .tile { opacity: 0.8; }
  input { width: 9rem; padding: 6px 8px; background: white; }
  button { padding: 6px 12px; background: white; cursor: pointer; }
  .players { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .player {
    background: white;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 10px 12px;
  }
  .player.turn { border-color: var(--ink); }
  .player h2 { margin: 0 0 6px; font-size: 15px; font-weight: 650; }
  .score { font-variant-numeric: tabular-nums; }
  .rack { display: flex; gap: 6px; min-height: 48px; margin-top: 8px; }
  .board-wrap { display: flex; justify-content: center; margin: 16px 0; }
  .board {
    display: grid;
    grid-template-columns: 28px repeat(5, 58px);
    grid-template-rows: 28px repeat(5, 58px);
    gap: 4px;
    background: var(--board);
    padding: 8px;
    border-radius: 12px;
  }
  .label {
    display: grid;
    place-items: center;
    color: var(--muted);
    font-size: 13px;
  }
  .square, .tile {
    display: grid;
    place-items: center;
    border-radius: 8px;
    font-size: 22px;
    font-weight: 700;
  }
  .square {
    background: #fbf7f0;
    border: 1px solid transparent;
    padding: 0;
  }
  .square.center { box-shadow: inset 0 0 0 2px #d2c3a4; }
  .square.open { cursor: pointer; }
  .square.open:hover { border-color: var(--ink); }
  .tile {
    width: 44px;
    height: 44px;
    border: 0;
    padding: 0;
  }
  .tile.held { outline: 2px solid var(--ink); outline-offset: 2px; }
  .tile.A, .square.A { background: #f4d35e; color: #2b2622; }
  .tile.B, .square.B { background: #ee6c4d; color: #2b2622; }
  .tile.C, .square.C { background: #3d5a80; color: white; }
  .tile.D, .square.D { background: #2a9d8f; color: #10211f; }
  .square.ghost { outline: 2px dashed var(--ink); outline-offset: -4px; }
  .status { min-height: 3em; margin: 0 0 12px; }
  .status.legal { color: var(--ok); }
  .status.illegal { color: var(--bad); }
  .actions { display: flex; gap: 8px; }
  .actions #play { background: var(--ink); color: white; border-color: var(--ink); }
  .meta { color: var(--muted); margin: 14px 0; }
  ol { margin: 0; padding-left: 1.2rem; }
  li { margin: 2px 0; }
  details { margin-top: 18px; color: var(--muted); }
</style>
</head>
<body>
<main>
  <header>
    <h1>Mini-Scrabble</h1>
    <form id="new-game">
      <input id="seed" inputmode="numeric" aria-label="Seed" placeholder="seed">
      <button type="submit">New game</button>
    </form>
  </header>
  <section class="players">
    <article class="player" id="player-0"></article>
    <article class="player" id="player-1"></article>
  </section>
  <div class="board-wrap"><div class="board" id="board"></div></div>
  <p class="status" id="status"></p>
  <div class="actions">
    <button id="pass" type="button">Pass</button>
    <button id="clear" type="button">Clear</button>
    <button id="play" type="button">Play</button>
  </div>
  <p class="meta" id="meta"></p>
  <ol id="log"></ol>
  <details>
    <summary>Rules used by this board</summary>
    <p>Place 1–3 tiles in one row or column. Gaps must already hold tiles. The first word uses at least two tiles and covers the center. Later words must touch an existing tile by an edge. Two-letter words are all legal. Three-letter words are all one letter or all different. Four-letter words use each letter once. Longer runs are illegal. The score is the sum of the lengths of the words you create or extend. Two passes in a row end the game, as does an empty bag together with an empty rack.</p>
  </details>
</main>
<script>
let game = null;
let hand = null;
let draft = [];
let flash = "";
let previewToken = 0;

const statusEl = document.getElementById("status");
const boardEl = document.getElementById("board");

function post(url, body) {
  return fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  }).then((response) => response.json());
}

function usedIndices() {
  return new Set(draft.map((tile) => tile.rackIndex));
}

function render() {
  for (const player of [0, 1]) {
    const card = document.getElementById("player-" + player);
    card.replaceChildren();
    card.classList.toggle("turn", player === game.player && !game.over);
    const title = document.createElement("h2");
    const moving = player === game.player && !game.over ? " to move" : "";
    title.textContent = "Player " + (player + 1) + moving;
    const score = document.createElement("div");
    score.className = "score";
    score.textContent = String(game.scores[player]);
    const rack = document.createElement("div");
    rack.className = "rack";
    const mine = player === game.player && !game.over;
    game.racks[player].forEach((letter, index) => {
      const tile = document.createElement("button");
      tile.type = "button";
      tile.className = "tile " + letter;
      tile.textContent = letter;
      const used = mine && usedIndices().has(index);
      tile.disabled = !mine || used;
      if (mine && hand === index) tile.classList.add("held");
      tile.addEventListener("click", () => hold(index));
      rack.append(tile);
    });
    card.append(title, score, rack);
  }

  boardEl.replaceChildren();
  const corner = document.createElement("div");
  boardEl.append(corner);
  for (let col = 1; col <= 5; col += 1) {
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = String(col);
    boardEl.append(label);
  }
  for (let row = 1; row <= 5; row += 1) {
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = String(row);
    boardEl.append(label);
    for (let col = 1; col <= 5; col += 1) {
      const square = document.createElement("button");
      square.type = "button";
      const ghost = draft.find((tile) => tile.row === row && tile.col === col);
      const letter = ghost ? ghost.letter : game.board[row - 1][col - 1];
      square.className = "square";
      if (letter !== ".") square.classList.add(letter);
      if (ghost) square.classList.add("ghost");
      if (row === 3 && col === 3) square.classList.add("center");
      square.textContent = letter === "." ? "" : letter;
      const empty = letter === "." || ghost;
      if (!game.over && empty) {
        square.classList.add("open");
        square.addEventListener("click", () => clickSquare(row, col));
      } else {
        square.disabled = true;
      }
      boardEl.append(square);
    }
  }

  document.getElementById("pass").disabled = game.over;
  document.getElementById("clear").disabled = draft.length === 0 && hand === null;
  document.getElementById("play").disabled = game.over || draft.length === 0;
  document.getElementById("meta").textContent =
    "Bag  " + game.bag.map((item) => item.letter + "×" + item.count).join("  ")
    + "    ·    seed " + game.seed
    + "    ·    " + game.legal + " legal actions";
  const log = document.getElementById("log");
  log.replaceChildren();
  for (const line of game.log) {
    const item = document.createElement("li");
    item.textContent = line;
    log.append(item);
  }
  showStatus();
  if (draft.length > 0) refreshPreview();
  else previewToken += 1;
}

function showStatus() {
  if (draft.length === 0) {
    statusEl.className = "status" + (flash ? " illegal" : "");
    statusEl.textContent = flash || game.hint;
    return;
  }
  statusEl.className = "status";
  statusEl.textContent = "Checking the placement…";
}

function hold(index) {
  flash = "";
  hand = hand === index ? null : index;
  render();
}

function clickSquare(row, col) {
  flash = "";
  const existing = draft.findIndex((tile) => tile.row === row && tile.col === col);
  if (existing >= 0) {
    draft.splice(existing, 1);
    hand = null;
    render();
    return;
  }
  if (hand === null) {
    flash = "Pick a tile from your rack first.";
    showStatus();
    return;
  }
  const letter = game.racks[game.player][hand];
  draft.push({letter, row, col, rackIndex: hand});
  hand = null;
  render();
}

function refreshPreview() {
  if (draft.length === 0) return;
  const token = ++previewToken;
  const cells = draft.map(({letter, row, col}) => ({letter, row, col}));
  post("/api/preview", {cells}).then((verdict) => {
    if (token !== previewToken) return;
    statusEl.className = "status " + verdict.status;
    statusEl.textContent = verdict.message;
  }).catch(() => {
    if (token !== previewToken) return;
    statusEl.className = "status illegal";
    statusEl.textContent = "Could not reach the game.";
  });
}

function apply(payload) {
  game = payload.game;
  if (payload.ok) {
    draft = [];
    hand = null;
    flash = "";
  } else {
    flash = payload.message;
  }
  render();
}

document.getElementById("clear").addEventListener("click", () => {
  draft = [];
  hand = null;
  flash = "";
  render();
});

document.getElementById("play").addEventListener("click", () => {
  const cells = draft.map(({letter, row, col}) => ({letter, row, col}));
  post("/api/play", {cells}).then(apply);
});

document.getElementById("pass").addEventListener("click", () => {
  post("/api/pass", {}).then(apply);
});

document.getElementById("new-game").addEventListener("submit", (event) => {
  event.preventDefault();
  const raw = document.getElementById("seed").value.trim();
  post("/api/new", {seed: raw}).then((payload) => {
    draft = [];
    hand = null;
    flash = payload.ok ? "" : payload.message;
    if (payload.game) game = payload.game;
    render();
  });
});

fetch("/api/state").then((response) => response.json()).then((payload) => {
  game = payload.game;
  document.getElementById("seed").value = String(game.seed);
  render();
});
</script>
</body>
</html>
"""


def _decode(length: int, code: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(LETTERS[code % N_LETTERS])
        code //= N_LETTERS
    return "".join(reversed(chars))


def _consider(length: int, code: int, words: list[str]) -> str | None:
    if length <= 1:
        return None
    word = _decode(length, code)
    if length > 4:
        return f"{word} is longer than 4."
    if not is_legal_word(word):
        return f"{word} is not a legal word."
    words.append(word)
    return None


def review(
    board: np.ndarray,
    rack: np.ndarray,
    opened: bool,
    cells: list[tuple[int, int, int]],
) -> dict[str, object]:
    """Say whether a placement is legal, and why.

    ``cells`` are ``(row, col, letter)`` with 0-based coordinates. The
    environment mask is still what a game accepts; this text is for the player.
    """
    idle = "Pick tiles from your rack and set them on the board."
    if not cells:
        return {"status": "idle", "score": 0, "message": idle, "words": []}
    if len(cells) > 3:
        return _illegal("You can place at most three tiles.")

    squares = [(row, col) for row, col, _ in cells]
    if len(set(squares)) != len(cells):
        return _illegal("Two tiles cannot share a square.")
    need = np.zeros(N_LETTERS, dtype=np.int64)
    for _, _, letter in cells:
        need[letter] += 1
    if np.any(need > np.asarray(rack)):
        return _illegal("Your rack does not have those tiles.")
    board_arr = np.asarray(board)
    for row, col, _ in cells:
        if int(board_arr[row * BOARD_SIZE + col]) != 0:
            return _illegal(f"({row + 1},{col + 1}) already has a tile.")

    rows = {row for row, _, _ in cells}
    cols = {col for _, col, _ in cells}
    if len(rows) != 1 and len(cols) != 1:
        return _illegal("The new tiles must lie in one row or one column.")
    if len(rows) == 1:
        row = next(iter(rows))
        placed = {col for _, col, _ in cells}
        for col in range(min(placed), max(placed) + 1):
            if col not in placed and int(board_arr[row * BOARD_SIZE + col]) == 0:
                return _illegal("An empty square is left between the new tiles.")
    else:
        col = next(iter(cols))
        placed = {row for row, _, _ in cells}
        for row in range(min(placed), max(placed) + 1):
            if row not in placed and int(board_arr[row * BOARD_SIZE + col]) == 0:
                return _illegal("An empty square is left between the new tiles.")

    if not opened:
        covers = any(row * BOARD_SIZE + col == CENTER for row, col, _ in cells)
        if len(cells) < 2 or not covers:
            return _illegal("The first placement must use at least two tiles and cover (3,3).")
    elif not _touches(board_arr, cells):
        return _illegal("A new tile must share an edge with a tile already on the board.")

    try:
        action = action_for(cells)
    except KeyError:
        return _illegal("That placement is not in the action table.")
    words: list[str] = []
    count = int(N_TILES[action])
    origin = int(SQUARES[action, 0])
    row, col = divmod(origin, BOARD_SIZE)
    horizontal = int(AXIS[action]) == 0
    reason = _consider(*_run(board_arr, SQUARES[action], LETTERS_AT[action], count, row, col, horizontal), words)
    if reason:
        return _illegal(reason)
    for slot in range(count):
        pos = int(SQUARES[action, slot])
        cross_row, cross_col = divmod(pos, BOARD_SIZE)
        reason = _consider(
            *_run(
                board_arr,
                SQUARES[action],
                LETTERS_AT[action],
                count,
                cross_row,
                cross_col,
                not horizontal,
            ),
            words,
        )
        if reason:
            return _illegal(reason)
    score = placement_score(board_arr, np.asarray(rack), opened, action)
    if score < 0 or score != sum(len(word) for word in words):
        return _illegal("That placement is illegal.")
    shown = " + ".join(f"{word} ({len(word)})" for word in words)
    return {"status": "legal", "score": int(score), "message": f"Legal. {shown} = {score}.", "words": words}


def _illegal(message: str) -> dict[str, object]:
    return {"status": "illegal", "score": 0, "message": message, "words": []}


def _touches(board: np.ndarray, cells: list[tuple[int, int, int]]) -> bool:
    for row, col, _ in cells:
        pos = row * BOARD_SIZE + col
        if row > 0 and int(board[pos - BOARD_SIZE]) != 0:
            return True
        if row < BOARD_SIZE - 1 and int(board[pos + BOARD_SIZE]) != 0:
            return True
        if col > 0 and int(board[pos - 1]) != 0:
            return True
        if col < BOARD_SIZE - 1 and int(board[pos + 1]) != 0:
            return True
    return False


def _rack_letters(state: np.ndarray, player: int) -> list[str]:
    letters = []
    base = RACK + player * N_LETTERS
    for index, name in enumerate(LETTERS):
        letters.extend([name] * int(state[base + index]))
    return letters


def _board_rows(state: np.ndarray) -> list[list[str]]:
    rows = []
    for row in range(BOARD_SIZE):
        cells = []
        for col in range(BOARD_SIZE):
            value = int(state[row * BOARD_SIZE + col])
            cells.append("." if value == 0 else LETTERS[value - 1])
        rows.append(cells)
    return rows


class Table:
    """One game, shared with the page."""

    def __init__(self, seed: int | None = None) -> None:
        self.env = MiniScrabbleEnv(1, seed=0)
        self.log: list[str] = []
        self.seed = 0
        self._lock = threading.RLock()
        self.new(seed)

    def new(self, seed: int | None) -> dict[str, object]:
        """Deal a game. ``None`` picks a seed."""
        with self._lock:
            chosen = secrets.randbelow(2**31) if seed is None else seed
            self.env.reset(seed=chosen)
            self.seed = chosen
            self.log = []
            player = int(self.env.state[0, PLAYER]) + 1
            self.log.append(f"New game, seed {chosen}. Player {player} to move.")
            return self.public(ok=True, message=self.hint())

    def preview(self, cells: list[tuple[int, int, int]]) -> dict[str, object]:
        with self._lock:
            if bool(self.env.terminated[0]):
                return _illegal("The game is over.")
            return review(*self._position(), cells)

    def play(self, cells: list[tuple[int, int, int]]) -> dict[str, object]:
        with self._lock:
            if bool(self.env.terminated[0]):
                return self.public(ok=False, message="The game is over.")
            verdict = review(*self._position(), cells)
            if verdict["status"] != "legal":
                return self.public(ok=False, message=str(verdict["message"]))
            action = action_for(cells)
            if not bool(self.env.mask[0, action]):
                return self.public(ok=False, message="That placement is illegal.")
            player = int(self.env.state[0, PLAYER]) + 1
            self.env.step(np.array([action], dtype=np.int64))
            reward = int(self.env.rewards[0])
            self.log.append(f"Player {player} played {action_name(action)} for {reward}.")
            self._note_result()
            return self.public(ok=True, message=self.hint())

    def pass_turn(self) -> dict[str, object]:
        with self._lock:
            if bool(self.env.terminated[0]):
                return self.public(ok=False, message="The game is over.")
            player = int(self.env.state[0, PLAYER]) + 1
            self.env.step(np.array([PASS], dtype=np.int64))
            self.log.append(f"Player {player} passed.")
            self._note_result()
            return self.public(ok=True, message=self.hint())

    def hint(self) -> str:
        result = self._result()
        if result is not None:
            return result
        player = int(self.env.state[0, PLAYER]) + 1
        passes = int(self.env.state[0, PASSES])
        trail = f" Consecutive passes: {passes}."
        if int(self.env.state[0, OPENED]) == 0:
            return (
                f"Player {player} to move. The first word needs at least two tiles "
                f"and must cover the center.{trail}"
            )
        return f"Player {player} to move.{trail}"

    def public(self, ok: bool, message: str) -> dict[str, object]:
        with self._lock:
            return self._snapshot(ok, message)

    def _snapshot(self, ok: bool, message: str) -> dict[str, object]:
        state = self.env.state[0]
        bag = [
            {"letter": letter, "count": int(state[BAG + index])}
            for index, letter in enumerate(LETTERS)
        ]
        payload = {
            "ok": ok,
            "message": message,
            "game": {
                "board": _board_rows(state),
                "racks": [_rack_letters(state, 0), _rack_letters(state, 1)],
                "bag": bag,
                "scores": [int(self.env.score[0, 0]), int(self.env.score[0, 1])],
                "player": int(state[PLAYER]),
                "passes": int(state[PASSES]),
                "opened": bool(state[OPENED]),
                "over": bool(self.env.terminated[0]),
                "hint": self.hint(),
                "log": list(self.log),
                "seed": self.seed,
                "legal": int(self.env.mask[0].sum()),
            },
        }
        return payload

    def _position(self) -> tuple[np.ndarray, np.ndarray, bool]:
        state = self.env.state[0]
        player = int(state[PLAYER])
        start = RACK + player * N_LETTERS
        rack = state[start : start + N_LETTERS].astype(np.int64)
        board = state[:N_SQUARES].astype(np.int64)
        return board, rack, bool(state[OPENED])

    def _result(self) -> str | None:
        if not bool(self.env.terminated[0]):
            return None
        first = int(self.env.score[0, 0])
        second = int(self.env.score[0, 1])
        line = f"{first}–{second}"
        if first > second:
            return f"Player 1 wins {line}."
        if second > first:
            return f"Player 2 wins {line}."
        return f"Draw {line}."

    def _note_result(self) -> None:
        result = self._result()
        if result is not None:
            self.log.append(result)


def parse_cells(data: object) -> tuple[list[tuple[int, int, int]] | None, str | None]:
    """Read ``{cells: [{letter, row, col}, ...]}`` using 1-based coordinates."""
    if not isinstance(data, dict):
        return None, "Expected a JSON object."
    raw = data.get("cells", [])
    if not isinstance(raw, list):
        return None, "Cells must be a list."
    cells = []
    for item in raw:
        if not isinstance(item, dict):
            return None, "Each tile needs a letter, row, and column."
        letter = item.get("letter")
        row = item.get("row")
        col = item.get("col")
        if not isinstance(letter, str) or letter not in LETTERS:
            return None, "Letters are A, B, C, and D."
        if isinstance(row, bool) or isinstance(col, bool):
            return None, "Rows and columns are 1–5."
        if not isinstance(row, int) or not isinstance(col, int) or not 1 <= row <= 5 or not 1 <= col <= 5:
            return None, "Rows and columns are 1–5."
        cells.append((row - 1, col - 1, LETTERS.index(letter)))
    return cells, None


def parse_seed(data: object) -> tuple[int | None, str | None]:
    if not isinstance(data, dict):
        return None, "Expected a JSON object."
    raw = data.get("seed", None)
    if raw is None or raw == "":
        return None, None
    if isinstance(raw, str) and raw.strip().isdigit():
        raw = int(raw.strip())
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0 or raw >= 2**64:
        return None, "Seed must be an integer from 0 up to 2^64-1."
    return raw, None


def make_server(host: str, port: int, table: Table) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, _PAGE, "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send(200, table.public(ok=True, message=table.hint()), "application/json")
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self._send(404, {"ok": False, "message": "Not found."}, "application/json")

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            data, error = self._read_json()
            if error is not None:
                self._send(400, {"ok": False, "message": error}, "application/json")
                return
            if path == "/api/new":
                seed, seed_error = parse_seed(data)
                if seed_error is not None:
                    self._send(400, {"ok": False, "message": seed_error}, "application/json")
                    return
                self._send(200, table.new(seed), "application/json")
                return
            cells, cell_error = parse_cells(data)
            if cell_error is not None or cells is None:
                self._send(400, {"ok": False, "message": cell_error}, "application/json")
                return
            if path == "/api/preview":
                self._send(200, table.preview(cells), "application/json")
            elif path == "/api/play":
                self._send(200, table.play(cells), "application/json")
            elif path == "/api/pass":
                self._send(200, table.pass_turn(), "application/json")
            else:
                self._send(404, {"ok": False, "message": "Not found."}, "application/json")

        def log_message(self, fmt: str, *args: object) -> None:
            return

        def _read_json(self) -> tuple[object, str | None]:
            length = self.headers.get("Content-Length", "0")
            if not length.isdigit():
                return None, "Missing a JSON body."
            raw = self.rfile.read(int(length))
            if not raw:
                return {}, None
            try:
                return json.loads(raw), None
            except json.JSONDecodeError:
                return None, "The body is not JSON."

        def _send(self, code: int, body: object, content_type: str) -> None:
            if isinstance(body, (dict, list)):
                data = json.dumps(body).encode()
            elif isinstance(body, str):
                data = body.encode()
            else:
                data = body
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer((host, port), Handler)


def _bind(host: str, port: int, table: Table) -> ThreadingHTTPServer:
    """Use ``port``, or the next free one if another board is already there."""
    for candidate in range(port, port + 20):
        try:
            return make_server(host, candidate, table)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
    raise SystemExit(f"Ports {port}–{port + 19} are already in use.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play Mini-Scrabble in a browser.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    table = Table(args.seed)
    server = _bind(args.host, args.port, table)
    url = f"http://{args.host}:{server.server_address[1]}"
    if server.server_address[1] != args.port:
        print(f"Port {args.port} is already in use.")
    print(f"Mini-Scrabble  {url}")
    if not args.no_browser:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
