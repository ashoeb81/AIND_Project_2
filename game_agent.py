"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import numpy as np


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def manhattan_distance(p1, p2):
    """Computes manhattan distance between two points.

    Parameters
    ----------
    p1: Tuple of coordinates (x,y).
    p2: Tuple of coordinates (x,y).

    Returns
    -------
    float
        The sum of absolute differences between the coordinates of p1 and p2.
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def weighted_move_difference(game, player):
    """Calculate the Weighted Move Difference (WMD) heuristic.

    See heuristic_analysis.pdf for details and analysis.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The Weighted Move Difference (WMD) heuristic value.
    """
    # Agent move count.
    own_moves = len(game.get_legal_moves(player))

    # Opponent move count.
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(2*own_moves - opp_moves)


def cornering(game, player):
    """Calculate the Cornering (C) heuristic.

    See heuristic_analysis.pdf for details and analysis.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The Cornering (C) heuristic value.
    """
    # Game board corners
    corners = [(0, 0), (0, game.width), (game.height, 0), (game.height, game.width)]

    # Agent minimum distance to a game board corner.
    own_pos = game.get_player_location(player)
    own_dist = min([manhattan_distance(own_pos, p) for p in corners])

    # Opponent minimum distance to a game board corner.
    opp_pos = game.get_player_location(game.get_opponent(player))
    opp_dist = min([manhattan_distance(opp_pos, p) for p in corners])

    return float(own_dist - opp_dist)


def space_access(game, player):
    """Calculate the Space Access (AS) heuristic.

    See heuristic_analysis.pdf for details and analysis.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The Space Access (AS) heuristic value.
    """
    # Game board corners
    blank_spaces = game.get_blank_spaces()

    # Agent average distance to a game board blank space.
    own_pos = game.get_player_location(player)
    own_dist = np.mean([manhattan_distance(own_pos, p) for p in blank_spaces])

    # Opponent average distance to a game board blank space.
    opp_pos = game.get_player_location(game.get_opponent(player))
    opp_dist = np.mean([manhattan_distance(opp_pos, p) for p in blank_spaces])

    return float(opp_dist - own_dist)


def opponent_pursuit(game, player):
    """Calculate the Opponent Pursuit (OP) heuristic.

    See heuristic_analysis.pdf for details and analysis.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The Opponent Pursuit (OP) heuristic value.
    """
    # Agent position
    own_pos = game.get_player_location(player)

    # Opponent position
    opp_pos = game.get_player_location(game.get_opponent(player))

    return manhattan_distance(own_pos, opp_pos)


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return 3*weighted_move_difference(game, player) + 2*cornering(game, player)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def check_timer(self):
        "Method to check whether there remains time to search."
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # If no moves available return.
        if len(legal_moves) == 0:
            return (-1, -1)

        try:
            # Determine which search method to use.
            if self.method == 'minimax':
                search_method = self.minimax
            else:
                search_method = self.alphabeta

            if self.iterative:
                # If iterative, then calculate best move for
                # increasing depths while time permits.
                depth = 1
                while(True):
                    _, best_move = search_method(game, depth)
                    depth += 1
            else:
                # Otherwise, calculate best move using specified search depth.
                _, best_move = search_method(game, self.search_depth)

        except Timeout:
            # Return the best move we found before the timeout.
            return best_move

        # Return the best move from the last completed search iteration.
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        def max_value(game, current_depth):
            """Maximizer Action"""
            self.check_timer()
            # If we reached specified depth, then evaluate node score.
            if current_depth == depth:
                return self.score(game, self)
            # If there are no moves left, see if we won or lost.
            moves = game.get_legal_moves(self)
            if len(moves) == 0:
                return game.utility(self)
            else:
                # Otherwise select action that yields the maximum score.
                v = float("-inf")
                for move in moves:
                    v = max(v, min_value(game.forecast_move(move), current_depth+1))
                return v


        def min_value(game, current_depth):
            """Minimizer Action."""
            self.check_timer()
            # If we reached specified depth, then evaluate node score.
            if current_depth == depth:
                return self.score(game, self)
            # If there are no moves left, see if we won or lost.
            moves = game.get_legal_moves(game.get_opponent(self))
            if len(moves) == 0:
                return game.utility(self)
            else:
                # Otherwise select action that yields the minimum score.
                v = float("inf")
                for move in moves:
                    v = min(v, max_value(game.forecast_move(move), current_depth+1))
                return v

        # Minimax search begins here at the maximizer level.  We select the move
        # that maximizes the score propagated all the way up to the root.
        moves = game.get_legal_moves(player=self)
        scores = map(lambda move: min_value(game.forecast_move(move), 1), moves)
        return max(zip(scores, moves), key=lambda rec: rec[0])

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        def max_value(game, alpha, beta, current_depth):
            "Maximizer Action"
            self.check_timer()
            # If we reached specified depth, then evaluate node score.
            if current_depth == depth:
                return self.score(game, self)
            moves = game.get_legal_moves(self)
            # If there are no moves left, see if we won or lost.
            if len(moves) == 0:
                return game.utility(self)
            else:
                # Otherwise select action that yields the maximum score.  If we find
                # a node with a score > beta (best score so far for minimizer), no need
                # to proceed (ie. we can prune) since the minimizer will never allow play
                # to reach this point.
                v = float("-inf")
                for move in moves:
                    v = max(v, min_value(game.forecast_move(move), alpha, beta, current_depth+1))
                    if v >= beta:
                        return v
                    alpha = max(v, alpha)
                return v


        def min_value(game, alpha, beta, current_depth):
            "Minimizer Action"
            self.check_timer()
            # If we reached specified depth, then evaluate node score.
            if current_depth == depth:
                return self.score(game, self)
            moves = game.get_legal_moves(game.get_opponent(self))
            # If there are no moves left, see if we won or lost.
            if len(moves) == 0:
                return game.utility(self)
            else:
                # Otherwise select action that yields the minimum score.  If we find
                # a node with a score < alpha (best score so far for maximizer), no need
                # to proceed (ie. we can prune) since the maximizer will never allow play
                # to reach this point.
                v = float("inf")
                for move in moves:
                    v = min(v, max_value(game.forecast_move(move), alpha, beta, current_depth+1))
                    if v <= alpha:
                        return v
                    beta = min(v, beta)
                return v

        # Alpha-Beta search begins here with alpha = -inf (worst score for maximizer) and
        # beta = + inf (worst score for minimizer).
        alpha = float("-inf")
        best_move = (-1, -1)
        beta = float("inf")
        # Select move that maximizes the score propagated all the way up to the root.
        for move in game.get_legal_moves(player=self):
            v = min_value(game.forecast_move(move), alpha, beta, 1)
            if v > alpha:
                alpha = v
                best_move = move
        return alpha, best_move

