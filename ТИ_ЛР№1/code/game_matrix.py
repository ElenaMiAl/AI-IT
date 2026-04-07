class GameMatrix:
    def __init__(self):
        self.matrix = [
            [10, 11, 16, 15, 2],
            [9, 7, 6, 17, 1],
            [3, 0, 19, 15, 4],
            [0, 15, 13, 10, 6]
        ]
        self.rows = 4  
        self.cols = 5  
    
    def get_matrix(self):
        return self.matrix
    
    def get_dimensions(self):
        return self.rows, self.cols
