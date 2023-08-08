"""
Starter code for a Sudoku solver, adapted from UMass CS 220.

To run from the command line: python3 starter.py

This will run the unit tests at the end of the file.
"""
import unittest

class Board:
    """Board(board_dict) is a class that represents a Sudoku board, where:

    - board_dict is a dictionary from (row, col) to a list of available values.
      The rows and columns are integers in the range 0 .. 8 inclusive. Each
      available value is an integer in the range 1 .. 9 inclusive."""


    def __init__(self, board_dict):
        """board_dict should be a dictionary, as described in the definition of
        the Board class."""
        self.board_dict = board_dict


    def __str__(self):
        """Prints the board nicely, arranging board_dict like a Sudoku board."""
        rows = [ ]
        for row in range(0, 9):
            col_strs = [ ]
            for col in range(0, 9): 
                val_str = ''.join(str(x) for x in self.board_dict[(row, col)])
                # Pad with spaces for alignment.
                val_str = val_str + (' ' * (9 - len(val_str)))
                col_strs.append(val_str)
                # Horizontal bar to separate boxes
                if col == 2 or col == 5:
                    col_strs.append('|')
            rows.append(' '.join(col_strs))
            # Vertical bar to separate boxes
            if row == 2 or row == 5:
                rows.append(93 * '-')
        return '\n'.join(rows)


    def copy(self):
        """Creates a deep copy of this board. Use this method to create a copy
        of the board before modifying the available values in any cell."""
        new_dict = { }
        for (k, v) in self.board_dict.items():
            new_dict[k] = v.copy()
        return Board(new_dict)


    def value_at(self, row, col):
        """Returns the value at the cell with the given row and column (zero
        indexed). Produces None if the cell has more than one available
        value."""
        # gets the list of value(s) in this cell
        values = self.board_dict[(row, col)]
        # return the value in that cell if there is only 1 value
        if len(values) == 1:
            return values[0]
        return None

    def place(self, row, col, value):
            """Places value at the given row and column.
            
            Eliminates value from the peers of (row, col) and recursively calls
            place on any cell that is constrained to exactly one value."""
            # remove value from all cells in peers, and if a peer has 1 value left, place it
            for cell in peers(row, col):
                
                if len(self.board_dict[cell]) != 0:
                    if value in self.board_dict[cell]: 
                        self.board_dict[cell].remove(value)
                    
                    # after removing value from cell's list of possible values,
                    # if 1 value is left, recursively apply place for that value
                        if len(self.board_dict[cell]) == 1:
                            self.place(cell[0], cell[1], self.board_dict[cell][0])
            # if there's only 1 value in (row, col) already, it has already been
            # placed and its peers filtered so just return
            if len(self.board_dict[(row, col)]) == 1:
                return
            self.board_dict[(row, col)] = [value]

    
    def next_boards(board):
        """Returns a list of boards that have one cell filled.

        Selects a cell that is maximally constrained: i.e., has a minimum
        number of available values.
        """
        # find cell that has least number of available values
        prevValues = 10  #initialize previous to a len greater than what's possible in a cell
        minValues = prevValues  #initialize min values list & its cell
        minValuesCell = (0, 0)
        for row in range(0, 9):
            for col in range(0, 9):
                
                currentValues = board.board_dict[(row, col)]
                # checks if cell not already placed by 1 value as well as other condition
                if (len(currentValues) != 1) and ((prevValues > len(currentValues))):
                    minValues = currentValues
                    minValuesCell = (row, col)
                    prevValues = len(currentValues) #update previous
        
        nextBoards = []
        # with the min possible values that was found, create all possible boards
        # for each number in that min list 
        for value in minValues:
            # make sure each next board is independent for each value
            # we place in this cell
            nextBoard = board.copy()
            nextBoard.place(minValuesCell[0], minValuesCell[1], value)  
            nextBoards.append(nextBoard)
        return nextBoards
        

    def is_solved(self):
        """Returns True if the board is fully solved, and there are no choices
        to make at any cell."""
        # for each cell in this board, if there is a cell with more than
        # one value, return false, otherwise after checking every cell, return true
        for row in range(0, 9):
            for col in range(0, 9):
                if len(self.board_dict[(row, col)]) != 1:
                    return False
        return True
    

    def is_unsolvable(self):
        """Returns True if the board is unsolvable."""
        # for each cell in this board if a cell is empty or doesn't have
        # any values, it's unsolvable
        for row in range(0, 9):
            for col in range(0, 9):
                if len(self.board_dict[(row, col)]) == 0:
                    return True
        return False

def parse(sudoku_string):
    """Parses a string of 81 digits and periods for empty cells into, produce
    a Board that represents the Sudoku board."""
    board_dict = {}
    for r in range(0,9):
        for c in range(0,9):
            board_dict[(r,c)] = [1,2,3,4,5,6,7,8,9]
            
    board = Board(board_dict)
    for i in range(81):
        if sudoku_string[i] != ".":
            board.place(i//9, i % 9, int(sudoku_string[i]))
    return board


def peers(row, col):
    """Returns the peers of the given row and column and in same square, as a list of tuples."""
    peers = []
    # for all cells in the same row and same col, except itself
    for a in range(0, 9):
        peers.append((row, a))
    for b in range(0, 9):
        if (b, col) not in peers:
            peers.append((b, col))
         
    # get the upper left corner cell of the square that the given (row, col) is in
    # then move 3 to the right and 3 down from there to get all cells in square
    (upmostR, leftmostC) = ((row//3 * 3), (col//3 * 3))
    for r in range(upmostR, upmostR + 3):
        for c in range(leftmostC, leftmostC + 3):
            if (r,c) not in peers:
                peers.append((r,c))
    peers.remove((row,col))
    return peers

def solve(board):
    """Recursively solve the board."""
    # base cases:
    if board.is_solved():
        return board
    
    if board.is_unsolvable():
        return None
    
    # general case: go through each board in possible next boards list
    # and find the first correct board
    for b in board.next_boards():
        nextB = solve(b)
        # only want to return the recursive call if next board is solvable
        if nextB is not None:
            return nextB 
    return None

def check_solution(board):
    """Check to see if the board represents a valid Sudoku solution."""
    # valid solution -> each digit appears only once in every row, col & square
    # only need to check one square in the board
    # for each cell value in that square, check its peers & check other values in square
    
    #check whether board is solvable first
    if solve(board) == None:
        return False
    
    prevValue = 10 # initializing previous value in cell to a value that won't ever be in cell
    for row in range(0, 9):
        for col in range(0, 9):
            currentValue = board.board_dict[(row, col)]
            for cell in peers(row, col):
                if (currentValue == board.board_dict[(cell[0], cell[1])]):
                    return False
            prevValue = currentValue
    return True



class SudokuTests(unittest.TestCase):

    def test_solved_puzzle(self):
        s = "853697421914238675762145893128563947475982136396471582581724369637859214249316758"
        board = parse(s)
        assert(board.value_at(0,0)==8)
        
    def test_solve_one(self):
        s = ".53697421914238675762145893128563947475982136396471582581724369637859214249316758"
        board = parse(s)
        assert(board.value_at(0,0)==8)

    def test_solve_third(self):
        s = "85.697421914238675762145893128563947475982136396471582581724369637859214249316758"
        board = parse(s)
        assert(board.value_at(0,2)==3)

    def test_empty_puzzle(self):
        s = "."*81
        board = parse(s)
        board_dict = { }
        for i in range(9):
            for j in range(9):
                board_dict[(i, j)] = [ 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
        assert(board.board_dict == board_dict)

    def test_easy1(self):
        s = "85....4.1......67...21....3..85....7...982...3....15..5....43...37......2.9....58"
        board = parse(s)
        solution = solve(board)
        print(solution)
        if check_solution(solution):
            print("correct board solution")
        else:
            print("incorrect")
        assert(solution is not None)

    def test_medium1(self):
        s = ".1.....2..3..9..1656..7...33.7..8..........89....6......6.254..9.5..1..7..3.....2"
        board = parse(s)
        solution = solve(board)
        print(solution)
        if check_solution(solution):
            print("correct board solution")
        else:
            print("incorrect")
        assert(solution is not None)

    def test_medium2(self):
        s = "2...8.3...6..7..84.3.5..2.9...1.54.8.........4.27.6...3.1..7.4.72..4..6...4.1...3"
        board = parse(s)
        solution = solve(board)
        print(solution)
        if check_solution(solution):
            print("correct board solution")
        else:
            print("incorrect")
        assert(solution is not None)
    
    def test_mine(self):
        s = ".8..9......7...5.9..34........641........82.......387.16......84...3..1........6."
        board = parse(s)
        solution = solve(board)
        print(solution)
        if check_solution(solution):
            print("correct board solution")
        else:
            print("incorrect")
        assert(solution is not None)


if __name__ == "__main__":
    unittest.main()    