import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class Neuron:

    def __init__(self, activation_type: str = 'threshold'):
       
        #  w = (w0, w1, w2, w3, w4)
        self.weights = np.zeros(5) # [0,0,0,0,0]
        self.act_type = activation_type
        self.loss_history = []  # История ошибок по эпохам
        self.weights_history = []  # История весов
        self.output_history = []  # История выходных векторов
        
    def activate(self, net: float) -> float:
        
        if self.act_type == 'threshold':
            # f(net) = { 1, net >= 0; 0, net < 0 } пороговая
            return 1.0 if net >= 0 else 0.0
        else:
            # f(net) = 1/2 * (net/(1+|net|) + 1) логическая
            return 0.5 * (net / (1 + abs(net)) + 1)
    
    def derivative(self, net: float) -> float:

        if self.act_type == 'threshold':
            # В случае НС с пороговой ФА коррекцию веса следует брать в виде Δw = η·δ·x
            return 1.0
        else:
            # Производная: df(net)/d net = 0.5/(1+|net|)^2
            return 0.5 / ((1 + abs(net)) ** 2)
    
    def forward(self, x: List[int], continuous: bool = False) -> float:
       
        # Добавляем x0 = 1 для смещения
        x_with_bias = [1.0] + [float(v) for v in x]
        
        #  net = Σ w_i * x_i + w_0
        net = sum(w * xi for w, xi in zip(self.weights, x_with_bias))
        
        # Применяем функцию активации 
        out = self.activate(net)
        
        if continuous:
            return out
      
        # y(out) = { 1, out >= 0.5; 0, out < 0.5 }
        return 1.0 if out >= 0.5 else 0.0
    
    def predict_all(self, X: List[List[int]]) -> List[int]:   # Предсказать выходы для всех входов
        return [int(self.forward(x)) for x in X]
    
    def train_step(self, x: List[int], target: int, eta: float) -> float: # Один шаг обучения по правилу Видроу-Хоффа
          
        # Прямой проход
        x_with_bias = [1.0] + [float(v) for v in x]
        net = sum(w * xi for w, xi in zip(self.weights, x_with_bias))
        out_cont = self.activate(net)
        output = 1.0 if out_cont >= 0.5 else 0.0
        
        # δ = t - y
        delta = target - output
        
        # Правило Видроу-Хоффа (дельта-правило)
        # w_i^(l+1) = w_i^(l) + Δw_i^(l)
        # Δw_i^(l) = η · δ^(l) · (df(net)/d net) · x_i^(l)
        if abs(delta) > 1e-6:
            df = self.derivative(net)  # Производная функции активации
            for i in range(5):
                self.weights[i] += eta * delta * df * x_with_bias[i]
        
        return abs(delta)
    
    def train_epoch(self, X: List[List[int]], targets: List[int], eta: float) -> Tuple[float, List[int]]:
        
        epoch_error = 0.0
        
        for x, t in zip(X, targets):
            epoch_error += self.train_step(x, t, eta)
        
        # После обновления весов вычисляем выходы для всех образцов
        outputs = self.predict_all(X)
        return epoch_error, outputs
    
    def train(self, X: List[List[int]], targets: List[int], 
              eta: float, max_epochs: int = 50, verbose: bool = False, 
              experiment_type: str = "full") -> List[float]:
      
        # На  каждой эпохе вычисляется суммарная квадратичная ошибка E(k)
      
        self.loss_history = []
        self.weights_history = [self.weights.copy()]
        self.output_history = []
        
        for epoch in range(max_epochs):
            error, outputs = self.train_epoch(X, targets, eta)
            self.loss_history.append(error)
            self.weights_history.append(self.weights.copy())
            self.output_history.append(outputs)
            
            if verbose:
                # Форматируем выходной вектор как в методичке (первые 8 значений)
                if experiment_type == "full" and len(outputs) > 8:
                    # Для полной выборки показываем первые 8 значений
                    first_eight = outputs[:16]
                    outputs_str = f"({', '.join(map(str, first_eight))}, ...)"
                else:
                    # Для минимального набора показываем все значения
                    outputs_str = f"({', '.join(map(str, outputs))})"
                
                w = self.weights
                print(f"Эпоха {epoch:2d}: ошибка = {error:.2f}, "
                      f"веса = ({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}, {w[3]:.2f}, {w[4]:.2f}), "
                      f"выход = {outputs_str}")
            
            if error == 0:
                if verbose:
                    print(f"Обучение завершено на эпохе {epoch}")
                break    
        return self.loss_history
    
    def print_weights_table(self, title: str = ""):
        
        if title:
            print(f"\n{title}")
        
        print("\nТаблица 1. Параметры НС на последовательных эпохах\n")
        print("| Номер эпохи k | Вектор весов w                    | Выходной вектор y                                | Суммарная ошибка E |\n")
        
        
        for epoch in range(len(self.loss_history)):
            w = self.weights_history[epoch]
            outputs = self.output_history[epoch]
            error = self.loss_history[epoch]
            
            w_str = f"({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}, {w[3]:.2f}, {w[4]:.2f})"
            
            if len(outputs) > 16:
                first_eight = outputs[:16]
                outputs_str = f"({', '.join(map(str, first_eight))})"
            else:
                outputs_str = f"({', '.join(map(str, outputs))})"
            
            print(f"| {epoch:13d} | {w_str:33s} | {outputs_str:29s} | {error:18.2f} |")
    
    def print_minimal_table(self, title: str = ""):
        
        if title:
            print(f"\n{title}")
        
        print("\nТаблица 3. Параметры НС на последовательных эпохах (минимальный набор)\n")
        print("| Номер эпохи k | Вектор весов w                    | Выходной вектор y                                | Суммарная ошибка E |\n")
        
        for epoch in range(len(self.loss_history)):
            w = self.weights_history[epoch]
            outputs = self.output_history[epoch]
            error = self.loss_history[epoch]
            
            w_str = f"({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}, {w[3]:.2f}, {w[4]:.2f})"
            outputs_str = f"({', '.join(map(str, outputs))})"
            
            print(f"| {epoch:13d} | {w_str:33s} | {outputs_str:18s} | {error:18.2f} |")

def boolean_function(x1: int, x2: int, x3: int, x4: int) -> int: # F = (¬x1 + ¬x2 + ¬x3) · (¬x2 + ¬x3 + x4)
    # Вычисляем отрицания
    not_x1 = 1 - x1
    not_x2 = 1 - x2
    not_x3 = 1 - x3
    
    # Первая дизъюнкция: ¬x1 + ¬x2 + ¬x3
    term1 = 1 if (not_x1 or not_x2 or not_x3) else 0
    
    # Вторая дизъюнкция: ¬x2 + ¬x3 + x4
    term2 = 1 if (not_x2 or not_x3 or x4) else 0
    
    # Конъюнкция (логическое умножение)
    return term1 and term2

def generate_truth_table() -> Tuple[List[List[int]], List[int]]: # Генерация таблицы истинности для всех 16 комбинаций
    X = []
    targets = []
    
    for i in range(16): # Раскладываем число на биты: x1, x2, x3, x4
        x1 = (i >> 3) & 1
        x2 = (i >> 2) & 1
        x3 = (i >> 1) & 1
        x4 = i & 1
        
        X.append([x1, x2, x3, x4])
        targets.append(boolean_function(x1, x2, x3, x4))
    return X, targets

def print_truth_table(X: List[List[int]], targets: List[int]):
    print("\nТАБЛИЦА ИСТИННОСТИ\n")
    print(" № | x1 x2 x3 x4 | ¬x1 ¬x2 ¬x3 | A=¬x1+¬x2+¬x3 | B=¬x2+¬x3+x4 | F \n")
    
    for i in range(16):
        x = X[i]
        not_x1 = 1 - x[0]
        not_x2 = 1 - x[1]
        not_x3 = 1 - x[2]
        
        term1 = 1 if (not_x1 or not_x2 or not_x3) else 0
        term2 = 1 if (not_x2 or not_x3 or x[3]) else 0
        f = targets[i]
        
        print(f"{i:2} | {x[0]}  {x[1]}  {x[2]}  {x[3]}   |  {not_x1}  {not_x2}  {not_x3}   |"
              f"      {term1}       |      {term2}       | {f}")
    
    zeros = sum(1 for t in targets if t == 0)
    ones = sum(1 for t in targets if t == 1)
    print("-"*100)

def run_experiment_full(activation_type: str, X: List[List[int]], targets: List[int], 
                        eta: float, max_epochs: int) -> Neuron:
   
    act_names = {'threshold': 'Пороговая', 'type2': 'Логическая'}
    
    print(f"\n1.{1 if activation_type == 'threshold' else 2}. Используя {act_names[activation_type]} ФА:\n")
   
    neuron = Neuron(activation_type)
    print(f"Начальные веса: ({neuron.weights[0]:.2f}, {neuron.weights[1]:.2f}, {neuron.weights[2]:.2f}, {neuron.weights[3]:.2f}, {neuron.weights[4]:.2f})")
    print(f"Норма обучения η = {eta}")
    print()
    neuron.train(X, targets, eta, max_epochs, verbose=True, experiment_type="full")
    print(f"\nИтоговые веса: ({neuron.weights[0]:.2f}, {neuron.weights[1]:.2f}, {neuron.weights[2]:.2f}, {neuron.weights[3]:.2f}, {neuron.weights[4]:.2f})")
    return neuron

def find_minimal_set_threshold(X: List[List[int]], targets: List[int]) -> Tuple[List[int], Neuron]: # Поиск минимального набора для пороговой функции
    
    print(f"\n2.1. Используя пороговую ФА.")
    print("-" * 70)
    
    # Для варианта 13 минимальный набор из 4 векторов, два представителя класса 0 (индексы 14, 15) и два представителя 1
    train_set = [14, 15, 0, 1]  # (1,1,1,0), (1,1,1,1), (0,0,0,0), (0,0,0,1)
    
    neuron = Neuron('threshold')
    train_X = [X[i] for i in train_set]
    train_t = [targets[i] for i in train_set]
    
    print(f"Минимальный набор из {len(train_set)} векторов:")
    for i, idx in enumerate(train_set):
        print(f"X^{({i+1})} = {train_X[i]}, t = {train_t[i]}")
    print()
    
    neuron.train(train_X, train_t, eta=0.3, max_epochs=20, verbose=True, experiment_type="min")
    print(f"\nДаёт следующие синаптические коэффициенты:")
    print(f"W = ({neuron.weights[0]:.2f}, {neuron.weights[1]:.2f}, {neuron.weights[2]:.2f}, {neuron.weights[3]:.2f}, {neuron.weights[4]:.2f})")
    print(f"Для полного обучения потребовалось {len(neuron.loss_history)} эпох.")
    return train_set, neuron

def find_minimal_set_type2(X: List[List[int]], targets: List[int]) -> Tuple[List[int], Neuron]: # Поиск минимального набора для логической функции
    
    print(f"\n2.2. Используя логическую функцию\n")
    # Для варианта 13 минимальный набор из 4 векторов, два представителя класса 0 (индексы 14, 15) и два представителя класса 1
    train_set = [14, 15, 0, 7]  # (1,1,1,0), (1,1,1,1), (0,0,0,0), (0,1,1,1)
    
    neuron = Neuron('type2')
    train_X = [X[i] for i in train_set]
    train_t = [targets[i] for i in train_set]
    
    print(f"Минимальный набор из {len(train_set)} векторов:")
    for i, idx in enumerate(train_set):
        print(f"X^{({i+1})} = {train_X[i]}, t = {train_t[i]}")
    print()
    neuron.train(train_X, train_t, eta=0.5, max_epochs=30, verbose=True, experiment_type="min")
    
    print(f"\nДаёт следующие синаптические коэффициенты:")
    print(f"W = ({neuron.weights[0]:.2f}, {neuron.weights[1]:.2f}, {neuron.weights[2]:.2f}, {neuron.weights[3]:.2f}, {neuron.weights[4]:.2f})")
    print(f"Для полного обучения потребовалось {len(neuron.loss_history)} эпох.")
    
    return train_set, neuron

def main():
    
    X, targets = generate_truth_table()  # Генерируем таблицу истинности
    print_truth_table(X, targets)
    
    print("\n1. ОБУЧЕНИЕ НС С ИСПОЛЬЗОВАНИЕМ ВСЕХ КОМБИНАЦИЙ ПЕРЕМЕННЫХ\n") 
    
    neuron1 = run_experiment_full('threshold', X, targets, eta=0.3, max_epochs=30)
    neuron2 = run_experiment_full('type2', X, targets, eta=0.5, max_epochs=30)
  
    print("\n2. ОБУЧЕНИЕ НС С ИСПОЛЬЗОВАНИЕМ ЧАСТИ КОМБИНАЦИЙ ПЕРЕМЕННЫХ\n")
    
    min_set1, neuron_min1 = find_minimal_set_threshold(X, targets)
    min_set2, neuron_min2 = find_minimal_set_type2(X, targets)
    
    print("\nТаблица 1. Параметры НС на последовательных эпохах (пороговая ФА)\n")
    neuron1.print_weights_table()
    
    print("\nТаблица 2. Параметры НС на последовательных эпохах (функция типа 2)\n")
    neuron2.print_weights_table()
    
    print("\nТаблица 3. Параметры НС на последовательных эпохах (пороговая ФА, минимальный набор)\n")
    neuron_min1.print_minimal_table()
    
    print("\nТаблица 4. Параметры НС на последовательных эпохах (функция типа 2, минимальный набор)\n")
    neuron_min2.print_minimal_table()
    
if __name__ == "__main__":
    main()
