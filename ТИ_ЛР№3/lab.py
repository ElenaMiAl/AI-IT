import numpy as np


class ContinuousGameSolver:
    
    def __init__(self, a, b, c, d, e):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        
    def H(self, x, y):
        return self.a*x**2 + self.b*y**2 + self.c*x*y + self.d*x + self.e*y
    
    def analytical_solution(self):
        A1, B1, C1 = 2*self.a, self.c, -self.d
        A2, B2, C2 = self.c, 2*self.b, -self.e
        
        det = A1*B2 - A2*B1
        x_star = (C1*B2 - C2*B1) / det
        y_star = (A1*C2 - A2*C1) / det
        
        x_star = max(0, min(1, x_star))
        y_star = max(0, min(1, y_star))
        
        v_star = self.H(x_star, y_star)
        
        print(f"\nЯдро игры: H(x,y) = {self.a}·x² + {self.b}·y² + {self.c}·x·y + {self.d}·x + {self.e}·y")
        print(f"\nОптимальная стратегия игрока A: x* = {x_star:.6f}")
        print(f"Оптимальная стратегия игрока B: y* = {y_star:.6f}")
        print(f"Цена игры: v = {v_star:.6f}")
        
        return x_star, y_star, v_star
    
    def create_payoff_matrix(self, N):
        x_points = np.linspace(0, 1, N+1)
        y_points = np.linspace(0, 1, N+1)
        
        matrix = np.zeros((N+1, N+1))
        for i, x in enumerate(x_points):
            for j, y in enumerate(y_points):
                matrix[i, j] = self.H(x, y)
        
        return matrix, x_points, y_points
    
    def solve_by_brown_robinson(self, matrix, max_iter=200, eps=1e-6):
        m, n = matrix.shape
        
        x_counts = np.zeros(m)
        y_counts = np.zeros(n)
        x_counts[0] = 1
        y_counts[0] = 1
        
        # Для отслеживания сходимости
        prev_x_star = None
        prev_y_star = None
        consecutive_small_errors = 0
        
        for k in range(1, max_iter + 1):
            x_emp = x_counts / k
            y_emp = y_counts / k
            
            expected_gains = np.dot(matrix, y_emp)
            best_i = np.argmax(expected_gains)
            
            expected_losses = np.dot(x_emp, matrix)
            best_j = np.argmin(expected_losses)
            
            x_counts[best_i] += 1
            y_counts[best_j] += 1
            
            # Проверка критерия остановки на каждой итерации
            x_star = x_counts / (k + 1)
            y_star = y_counts / (k + 1)
            
            x_idx = np.argmax(x_star)
            y_idx = np.argmax(y_star)
            
            if prev_x_star is not None and prev_y_star is not None:
                error = abs(x_star[x_idx] - prev_x_star[prev_x_idx]) + abs(y_star[y_idx] - prev_y_star[prev_y_idx])
                
                if error < eps:
                    consecutive_small_errors += 1
                    if consecutive_small_errors >= 5:
                        print(f"  Остановка на итерации {k}: ошибка < {eps} в течение 5 итераций подряд")
                        return x_idx, y_idx, k
                else:
                    consecutive_small_errors = 0
            
            prev_x_star = x_star.copy()
            prev_y_star = y_star.copy()
            prev_x_idx = x_idx
            prev_y_idx = y_idx
        
        print(f"  Достигнут максимум итераций ({max_iter})")
        return x_idx, y_idx, max_iter
    
    def numerical_solution(self, max_N=12):
        results = []
        
        for N in range(2, max_N + 1):
            matrix, x_points, y_points = self.create_payoff_matrix(N)
            
            # Вывод всей матрицы
            print(f"\nN={N}")
            print("[")
            for i in range(N+1):
                row_str = "["
                for j in range(N+1):
                    row_str += f"{matrix[i,j]:7.3f}"
                    if j < N:
                        row_str += " "
                row_str += "]"
                print(row_str)
            print("]")
            
            # Поиск седловой точки
            row_mins = np.min(matrix, axis=1)
            col_maxs = np.max(matrix, axis=0)
            lower_price = np.max(row_mins)
            upper_price = np.min(col_maxs)
            
            saddle_found = False
            saddle_i, saddle_j = -1, -1
            
            for i in range(N+1):
                for j in range(N+1):
                    if abs(matrix[i,j] - row_mins[i]) < 1e-6 and abs(matrix[i,j] - col_maxs[j]) < 1e-6:
                        if abs(lower_price - upper_price) < 1e-6:
                            saddle_found = True
                            saddle_i, saddle_j = i, j
                            break
                if saddle_found:
                    break
            
            if saddle_found:
                x_num = x_points[saddle_i]
                y_num = y_points[saddle_j]
                v_num = matrix[saddle_i, saddle_j]
                print(f"\nЕсть седловая точка:")
                print(f"x={x_num:.3f} y={y_num:.3f} H={v_num:.3f}")
                results.append({
                    'N': N, 'method': 'седловая точка',
                    'x': x_num, 'y': y_num, 'v': v_num, 'iterations': None
                })
            else:
                x_idx, y_idx, iterations = self.solve_by_brown_robinson(matrix)
                x_num = x_points[x_idx]
                y_num = y_points[y_idx]
                v_num = matrix[x_idx, y_idx]
                print(f"\nСедловой точки нет, решение методом Брауна-Робинсона:")
                print(f"x={x_num:.3f} y={y_num:.3f} H={v_num:.3f}")
                print(f"  Потребовалось итераций: {iterations}")
                results.append({
                    'N': N, 'method': 'Брауна-Робинсона',
                    'x': x_num, 'y': y_num, 'v': v_num, 'iterations': iterations
                })
        
        return results


def main():
    a, b, c, d, e = -4, 2, 8, -4/5, -32/5
    
    solver = ContinuousGameSolver(a, b, c, d, e)
    
    x_ana, y_ana, v_ana = solver.analytical_solution()
    
    numerical_results = solver.numerical_solution(max_N=5)
    
    print(f"\n{'='*60}")
    print(f"Аналитическое решение:    x* = {x_ana:.4f}, y* = {y_ana:.4f}, v = {v_ana:.4f}")
    
    best = numerical_results[-1]
    print(f"Численное решение (N={best['N']}): x* = {best['x']:.4f}, y* = {best['y']:.4f}, v = {best['v']:.4f}")
    if best['iterations']:
        print(f"  Итераций потребовалось: {best['iterations']}")
    print(f"{'='*60}")
    

if __name__ == "__main__":
    main()
