"""The chess rules (published perft counts, notation, endings) and the engine (lightwork.chess)."""

from __future__ import annotations

import random
import subprocess
import sys

import pytest

from lightwork.chess import Board, best_move, book_reply, parse_uci, perft, reply_to, square, uci


# Counts from the Chess Programming Wiki's "Perft Results": any wrong castling, en passant, promotion or pin shows up here.
@pytest.mark.parametrize("fen,counts", [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", (20, 400, 8902, 197281)),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", (48, 2039, 97862)),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", (14, 191, 2812, 43238)),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", (6, 264, 9467)),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", (44, 1486, 62379)),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", (46, 2079, 89890)),
])
def test_perft_matches_the_published_counts(fen, counts):
    board = Board(fen)
    for depth, want in enumerate(counts, 1):
        assert perft(board, depth) == want, depth
    assert board.fen().split()[:4] == fen.split()[:4]  # every push came off again


def test_undoing_moves_restores_key_and_score():
    board = Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
    key, score = board.key, board.score
    for m in board.legal_moves():
        board.push(m)
        fresh = Board(board.fen())
        assert fresh.key == board.key and fresh.score == board.score, uci(m)
        board.pop()
    assert (board.key, board.score) == (key, score)


@pytest.mark.parametrize("fen,move,text", [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "g1f3", "Nf3"),
    ("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "e4d5", "exd5"),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", "O-O"),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1", "O-O-O"),
    ("8/P6k/8/8/8/8/8/K7 w - - 0 1", "a7a8q", "a8=Q"),
    ("6k1/5ppp/8/8/8/8/8/K2R4 w - - 0 1", "d1d8", "Rd8#"),
    ("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "a1a8", "Ra8+"),
    ("4k3/8/8/8/8/5N2/8/1N2K3 w - - 0 1", "b1d2", "Nbd2"),
    ("4k3/8/8/8/1N6/8/1N6/4K3 w - - 0 1", "b2d3", "N2d3"),
    ("k7/8/8/2Q1Q3/8/2Q5/8/4K3 w - - 0 1", "c5d4", "Qc5d4"),
])
def test_notation(fen, move, text):
    assert Board(fen).san(parse_uci(move)) == text


def test_what_a_move_changes_on_a_real_board():
    castle = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert set(castle.changes(parse_uci("e1g1"))) == {square(s) for s in ("e1", "g1", "h1", "f1")}
    assert set(castle.changes(parse_uci("e1c1"))) == {square(s) for s in ("e1", "c1", "a1", "d1")}
    passing = Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert set(passing.changes(parse_uci("e5d6"))) == {square(s) for s in ("e5", "d6", "d5")}
    plain = Board()
    assert set(plain.changes(parse_uci("e2e4"))) == {square("e2"), square("e4")}


def test_the_endings():
    assert Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").outcome() == ("stalemate", 0)
    assert Board("R5k1/5ppp/8/8/8/8/8/K7 b - - 0 1").outcome() == ("checkmate", 1)
    assert Board("8/8/8/4k3/8/4K3/8/8 w - - 0 1").outcome() == ("too few pieces", 0)
    assert Board("8/8/8/4k3/8/4K3/4R3/8 w - - 100 80").outcome() == ("the fifty-move rule", 0)
    board = Board()
    for _ in range(2):
        for text in ("g1f3", "g8f6", "f3g1", "f6g8"):
            board.push(parse_uci(text))
    assert board.outcome() == ("repetition", 0)
    assert Board().outcome() is None


def test_passing_square_counts_only_when_a_pawn_can_take():
    board = Board()
    board.push(parse_uci("e2e4"))
    assert board.ep == -1  # nothing can take there
    board = Board("4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1")
    board.push(parse_uci("e2e4"))
    assert board.ep == square("e3")


@pytest.mark.parametrize("fen,want", [
    ("6k1/5ppp/8/8/8/8/8/K2R4 w - - 0 1", "d1d8"),  # mate in one
    ("rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3", None),  # any sensible move: must not hang the board
    ("3qk3/8/8/8/8/8/8/3QK3 w - - 0 1", "d1d8"),  # a free queen
])
def test_engine_finds_it(fen, want):
    board = Board(fen)
    before = board.fen()
    m = best_move(board, 0.8, random.Random(1))
    assert board.fen() == before
    assert m in board.legal_moves()
    if want:
        assert uci(m) == want


def test_engine_avoids_stalemating_when_winning():
    board = Board("7k/8/5K2/6Q1/8/8/8/8 w - - 0 1")
    m = best_move(board, 0.8, random.Random(2))
    board.push(m)
    assert board.outcome() != ("stalemate", 0)
    assert board.in_check() or board.legal_moves()


def test_book_follows_the_game_and_stops_where_it_ends():
    rng = random.Random(3)
    assert book_reply(["e2e4"], rng) in ("e7e5", "c7c5", "e7e6", "c7c6")
    assert book_reply(["a2a3"], rng) is None


def test_reply_after_leaving_the_book_is_legal():
    played = ["a2a3", "e7e5", "h2h3"]
    text = reply_to(played, 0.3, seed=5)
    board = Board()
    for t in played:
        board.push(parse_uci(t))
    assert parse_uci(text) in board.legal_moves()


def test_engine_runs_as_its_own_process():
    out = subprocess.run([sys.executable, "-m", "lightwork.chess", "--think", "0.2", "--seed", "1", "a2a3"],
                         capture_output=True, text=True, timeout=20)
    assert out.returncode == 0, out.stderr
    assert parse_uci(out.stdout.strip()) in _after(["a2a3"]).legal_moves()


def _after(played):
    board = Board()
    for t in played:
        board.push(parse_uci(t))
    return board


def test_self_play_stays_legal_and_ends():
    """A short game against itself: every position stays consistent and nothing crashes."""
    board = Board()
    rng = random.Random(4)
    for _ in range(24):
        if board.outcome() is not None:
            break
        m = best_move(board, 0.05, rng)
        assert m in board.legal_moves()
        board.san(m)
        board.spoken(m)
        board.push(m)
        assert Board(board.fen()).key == board.key
