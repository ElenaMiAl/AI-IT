import numpy as np

class GameSolver:
    def __init__(self, game):
        self.game = game
        self.m = game.rows
        self.n = game.cols
        self.matrix = game.matrix
    
    def solve_A(self):
       
        m, n = self.m, self.n
        table = self._build_table_A()
        rows, cols = table.shape
        basis = [f"s{i+1}" for i in range(n)]
        Cb = [0] * n
        
       
        table, basis, Cb = self._simplex_A(table, basis, Cb, rows, cols)
        
        u = self._extract_u(table, basis, m)
        W = table[-1][-1]          
        return u, W
    
    def _build_table_A(self):
        m, n = self.m, self.n
        table = np.zeros((n + 1, m + n + 1))
        
        
        for i in range(n):                    
            for j in range(m):                
                table[i][j] = -self.matrix[j][i]   
            table[i][m + n] = -1              
            table[i][m + i] = 1               
        
        for j in range(m):
            table[n][j] = 1                   
        
        return table
    
    def _simplex_A(self, table, basis, Cb, rows, cols):
        m, n = self.m, self.n
        
        for _ in range(100):   
           
            neg_rows = [i for i in range(n) if table[i][-1] < -1e-8]
            if neg_rows:
                
                pivot_row = min(neg_rows, key=lambda i: table[i][-1])
                
                pivot_col = -1
                for j in range(cols - 1):
                    if table[pivot_row][j] < -1e-8:
                        pivot_col = j
                        break
                if pivot_col == -1:
                    break   
            else:
               
                deltas = []
                for j in range(cols - 1):
                    s = sum(Cb[i] * table[i][j] for i in range(n))
                    Cj = 1 if j < m else 0
                    deltas.append(s - Cj)
                
               
                pos = [j for j, d in enumerate(deltas) if d > 1e-8]
                if not pos:
                    break   
                pivot_col = max(pos, key=lambda j: deltas[j])
                
                
                min_ratio = float('inf')
                pivot_row = -1
                for i in range(n):
                    if table[i][pivot_col] > 1e-8:
                        ratio = table[i][-1] / table[i][pivot_col]
                        if ratio < min_ratio:
                            min_ratio = ratio
                            pivot_row = i
                if pivot_row == -1:
                    break   
            
           
            table, basis, Cb = self._pivot_operation(
                table, pivot_row, pivot_col, basis, Cb, rows, m, n, 'A'
            )
        
        return table, basis, Cb
    
    def solve_B(self):
       
        m, n = self.m, self.n
        table = self._build_table_B()
        rows, cols = table.shape
        basis = [f"r{j+1}" for j in range(m)]
        Cb = [0] * m
        
        table, basis, Cb = self._simplex_B(table, basis, Cb, rows, cols)
        
        t = self._extract_t(table, basis, n)
        T = -table[-1][-1]        # значение целевой функции (V)
        v = 1.0 / T
        y = t * v
        return y, v
    
    def _build_table_B(self):
        m, n = self.m, self.n
        
        table = np.zeros((m + 1, n + m + 1))
        
        for j in range(m):              
            for i in range(n):            
                table[j][i] = self.matrix[j][i]
            table[j][n + j] = 1           
            table[j][-1] = 1              # 
        
        for i in range(n):
            table[m][i] = -1
        return table
    
    def _simplex_B(self, table, basis, Cb, rows, cols):
        m, n = self.m, self.n
        
        for _ in range(100):
           
            deltas = []
            for j in range(cols - 1):
                s = sum(Cb[i] * table[i][j] for i in range(m))
                cj = -1 if j < n else 0
                deltas.append(s - cj)
            
            pos = [j for j, d in enumerate(deltas) if d > 1e-8]
            if not pos:
                break
            
            pivot_col = max(pos, key=lambda j: deltas[j])
            
            min_ratio = float('inf')
            pivot_row = -1
            for i in range(m):
                if table[i][pivot_col] > 1e-8:
                    ratio = table[i][-1] / table[i][pivot_col]
                    if ratio < min_ratio:
                        min_ratio = ratio
                        pivot_row = i
            if pivot_row == -1:
                break
            
            table, basis, Cb = self._pivot_operation(
                table, pivot_row, pivot_col, basis, Cb, rows, m, n, 'B'
            )
        
        return table, basis, Cb
    
    def _pivot_operation(self, table, pivot_row, pivot_col, basis, Cb, rows, m, n, problem_type):
    
        elem = table[pivot_row][pivot_col]
        table[pivot_row] = table[pivot_row] / elem
        
        if problem_type == 'A':
            if pivot_col < m:
                basis[pivot_row] = f"u{pivot_col+1}"
                Cb[pivot_row] = 1
            else:
                basis[pivot_row] = f"s{pivot_col - m + 1}"
                Cb[pivot_row] = 0
        else:  # 'B'
            if pivot_col < n:
                basis[pivot_row] = f"t{pivot_col+1}"
                Cb[pivot_row] = -1
            else:
                basis[pivot_row] = f"r{pivot_col - n + 1}"
                Cb[pivot_row] = 0
        
        for i in range(rows):
            if i != pivot_row:
                factor = table[i][pivot_col]
                if abs(factor) > 1e-10:
                    table[i] = table[i] - factor * table[pivot_row]
        
        return table, basis, Cb
    
    def _extract_u(self, table, basis, m):
        u = np.zeros(m)
        for j in range(m):
            name = f"u{j+1}"
            if name in basis:
                idx = basis.index(name)
                u[j] = table[idx][-1]
        return u
    
    def _extract_t(self, table, basis, n):
        t = np.zeros(n)
        for i in range(n):
            name = f"t{i+1}"
            if name in basis:
                idx = basis.index(name)
                t[i] = table[idx][-1]
        return t
