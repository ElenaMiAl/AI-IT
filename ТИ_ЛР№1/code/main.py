from game_matrix import GameMatrix
from simplex_solver import GameSolver

def main():
    game = GameMatrix()
    solver = GameSolver(game)
    
    u, W = solver.solve_A()
    v_A = 1.0 / W
    x = u * v_A
    
    y, v_B = solver.solve_B()
    
    print("=" * 50)
    print("ВАРИАНТ 13")
    print("=" * 50)
    print(f"\nЦена игры (из A): v = {v_A:.6f}")
    print(f"Цена игры (из B): v = {v_B:.6f}\n")
    print("Стратегия A (игрок, минимизирующий):")
    for i in range(game.rows):
        print(f"  x{i+1} = {x[i]:.6f}")
    print("\nСтратегия B (игрок, максимизирующий):")
    for i in range(game.cols):
        print(f"  y{i+1} = {y[i]:.6f}")

if __name__ == "__main__":
    main()
