import sys

def print_grid(grid):
    """Display Sudoku grid in a readable format with borders."""
    print("\nSolved Sudoku:\n")
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)  # Horizontal separator
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")  # Vertical separator
            print(grid[i][j] if grid[i][j] != 0 else ".", end=" ")
        print()

def is_valid(grid, row, col, num):
    """Check if placing 'num' in grid[row][col] is valid."""
    for i in range(9):
        if grid[row][i] == num or grid[i][col] == num:
            return False
    
    start_row, start_col = (row // 3) * 3, (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if grid[start_row + i][start_col + j] == num:
                return False
    return True

def solve_sudoku(grid):
    """Solve Sudoku using backtracking."""
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                for num in range(1, 10):
                    if is_valid(grid, row, col, num):
                        grid[row][col] = num
                        if solve_sudoku(grid):
                            return True
                        grid[row][col] = 0
                return False
    return True

def get_sudoku_input():
    """Allow the user to input a Sudoku puzzle easily."""
    print("Enter Sudoku puzzle row by row (use 0 for empty cells):")
    grid = []
    for i in range(9):
        while True:
            row = input(f"Row {i+1}: ")
            row_vals = [int(x) for x in row.split() if x.isdigit()]
            if len(row_vals) == 9:
                grid.append(row_vals)
                break
            else:
                print("Invalid input. Please enter 9 numbers separated by spaces.")
    return grid

def main():
    print("Sudoku Solver - Enter your puzzle or type 'exit' to quit.")
    
    while True:
        print("\nOptions:")
        print("1. Enter Sudoku manually")
        print("2. Load Sudoku from a file")
        print("3. Quit")
        
        choice = input("Select an option: ")
        if choice == "1":
            grid = get_sudoku_input()
        elif choice == "2":
            filename = input("Enter file name: ")
            try:
                with open(filename, 'r') as f:
                    grid = [[int(num) for num in line.split()] for line in f.readlines()]
            except Exception as e:
                print(f"Error reading file: {e}")
                continue
        elif choice == "3":
            sys.exit("Goodbye!")
        else:
            print("Invalid option. Try again.")
            continue
        
        print("\nInitial Sudoku Puzzle:")
        print_grid(grid)
        confirm = input("Proceed with solving? (y/n): ")
        if confirm.lower() != 'y':
            continue
        
        if solve_sudoku(grid):
            print_grid(grid)
        else:
            print("No solution exists!")

if __name__ == "__main__":
    main()

