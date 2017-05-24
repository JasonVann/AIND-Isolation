"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

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

    my_legal_moves = game.get_legal_moves(player)
    opp = game.get_opponent(player)
    opp_legal_moves = game.get_legal_moves(opp)
    # return float(len(my_legal_moves) - 2 * len(opp_legal_moves))
    #  + float(len(my_legal_moves) * (10 / (1+len(opp_legal_moves)))) # -2%

    # A combination from heuristic 2 and 3
    return float(len(my_legal_moves) - 1.5 * len(opp_legal_moves)) + float(
        len(my_legal_moves) * (10 / (1.5 + len(opp_legal_moves))))  # + 7%


def custom_score_2(game, player):
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
    # If the opponent has too many legal moves, we want to penalize the agent with a multiplier of a

    my_legal_moves = game.get_legal_moves(player)
    opp = game.get_opponent(player)
    opp_legal_moves = game.get_legal_moves(opp)
    # return float(len(my_legal_moves) * (3 / (1+len(opp_legal_moves)))) #-3%
    # return float(len(my_legal_moves) - 0.5 * len(opp_legal_moves))  # +3-5%

    return float(len(my_legal_moves) - 2.5 * len(opp_legal_moves))  # +3-5%


def custom_score_3(game, player):
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
    # Try divide our number of legal moves by opponent's
    # Add 1 in the denominator to avoid division by 1 error

    my_legal_moves = game.get_legal_moves(player)
    opp = game.get_opponent(player)
    opp_legal_moves = game.get_legal_moves(opp)

    # return float(len(my_legal_moves) * (10 - len(opp_legal_moves)))
    # return float(len(my_legal_moves) * (len(my_legal_moves) - len(opp_legal_moves))) # 56.7%
    # return float(len(my_legal_moves) - 1 * len(opp_legal_moves))
    # + float(len(my_legal_moves) * (10 / (1 + len(opp_legal_moves))))  # + 1 - 2%
    # return float(len(my_legal_moves) * (1.5 / (1 + len(opp_legal_moves))))  # +3%

    return float(len(my_legal_moves) * (2 / (1 + len(opp_legal_moves))))  # +4%


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

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

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        actions = game.get_legal_moves()
        res = [(a, self.max_min(game.forecast_move(a), depth-1, False)) for a in actions]
        if not res:
            # []
            action = (-1, -1)
        else:
            max_obj = max(res, key=lambda x: x[1])  # Find max based on the max value
            action = max_obj[0]
        return action

        # Alternatively, use for loop instead of the above list comprehension
        # best_action = (-1, -1)
        # best_score = -float("inf")
        # for action in actions:
        #     # For each legal moves, forecast the board state and apply min_value for each
        #     score = self.min_value(game.forecast_move(action), depth - 1)
        #     if score > best_score:
        #         best_action = action
        #         best_score = score
        #
        # return best_action

    def max_min(self, game, depth, is_max = True):
        """
        A search method for the max or min layer, depending on the flag is_max
        Args:
            game : isolation.Board
                An instance of the Isolation game `Board` class representing the
                current game state
            depth : int
                Depth is an integer representing the maximum number of plies to
                search in the game tree before aborting
            is_max:
                If is_max is True, it's the max layer, and takes the max of its children
                If False, it's on the min layer, and takes the min of its children
                Either case, prune the tree if alpha >= beta
        Returns:
            the best score
        Raises:
            SearchTime()
        """

        if depth <= 0:
            return self.score(game, self)

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if is_max:
            value = - float("inf")
        else:
            value = float("inf")

        actions = game.get_legal_moves()

        # If there're no legal moves, then the following loop will fall through
        # and return -inf directly
        for action in actions:
            # Find the forcasted resulting state when action is applied to the given game state
            result = game.forecast_move(action)
            if is_max:
                value = max(value, self.max_min(result, depth - 1, False))
            else:
                value = min(value, self.max_min(result, depth - 1, True))
        return value


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

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

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # Iterative-Deepening: increase the search depth by 1 every iteration
            depth = self.search_depth
            while self.time_left() > self.TIMER_THRESHOLD:
                best_move = self.alphabeta(game, depth)
                depth += 1

        except SearchTimeout:
            return best_move  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

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

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        actions = game.get_legal_moves()
        best_action = (-1, -1)
        best_score = -float("inf")
        for action in actions:
            # For each legal action, we forecast the new state and try alpha-beta pruning
            # Use the best_score instead of "-inf" so that later iterations of the loop
            # will take advantage of the improved alpha
            score = self.max_min(game.forecast_move(action), depth-1, best_score, beta, False)
            if score > best_score:
                best_action = action
                best_score = score

        return best_action

    def max_min(self, game, depth, alpha=float("-inf"), beta=float("inf"), is_max=True):
        """
        An alpha-beta pruning method for the max or min layer, depending on the flag is_max
        Args:
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
            is_max:
                If True, it's on the max layer, and takes the max of alpha and children
                If False, it's on the min layer, and takes the min of the beta and children
                Either case, prune the tree if alpha >= beta
        Returns:
            the best score
        Raises:
            SearchTime()
        """

        if depth <= 0:
            return self.score(game, self)

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        actions = game.get_legal_moves()

        if is_max:
            for action in actions:
                # Find the forecast resulting state when action is applied to the given game state
                result = game.forecast_move(action)

                alpha = max(alpha, self.max_min(result, depth - 1, alpha, beta, False))
                if alpha >= beta:
                    break

            return alpha

        if not is_max:
            for action in actions:
                result = game.forecast_move(action)
                beta = min(beta, self.max_min(result, depth - 1, alpha, beta, True))

                if beta <= alpha:
                    break

            return beta
