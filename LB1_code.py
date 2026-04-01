import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import sys


def boolean_function(x1, x2, x3, x4):
    return 1 if (not (x2 and x3)) or ((not x1) and x4) else 0
   
n = 4
X, targets = [], []

for i in range(16):
    x1 = (i >> 3) & 1  
    x2 = (i >> 2) & 1  
    x3 = (i >> 1) & 1  
    x4 = (i >> 0) & 1  
    
    t = boolean_function(x1, x2, x3, x4)
    X.append([x1, x2, x3, x4])
    targets.append(t)
    
    x_str = f"{x1} {x2} {x3} {x4}"
    print(f"{i+1:2} | {x_str} | {t}")


def activate(net, type='1'):
    if type == '1':
        return 1.0 if net >= 0 else 0.0
    else:
        return 1.0 if net >= 0.5 else 0.0


def sigma(net):
    return 0.5 * (net / (1 + abs(net)) + 1)

def dsigma(net):
    return 1.0 / (2.0 * (1 + abs(net)) ** 2)
                          
    
def train_neuron(type):
    w = np.zeros(5)  
    error_history = np.zeros(50)
    y_net = np.zeros(16)
    e = 0
    delta = np.zeros(5)
    
    for ephoch in range(29):
        
        print("\nЭпоха", ephoch)
        print("\n")
        for i in range(16):
            net = round((w[0] + w[1]*X[i][0] + w[2]*X[i][1] + w[3]*X[i][2] + w[4]*X[i][3]), 3)
            if type == '2':
                net = sigma(net)
            y = activate(net, type)
            y_net[i] = y
            t = targets[i]
            error = t - y
            if error != 0:
                error_history[e] = error_history[e] + 1

                for j in range(5):
                    
                    if j == 0:
                        if type == '1':
                            delta[0] = delta[0] + 1 * 0.3 * error
                        else:
                            delta[0] = round((delta[0] + 1 * 0.3 * error * dsigma(net)), 4)
                    else:
                        if type == '1':
                            delta[j] = delta[j] + 0.3 * error * X[i][j-1]
                        else:
                            delta[j] = round((delta[j] + 0.3 * error * X[i][j-1] * dsigma(net)), 4)
                    
            w = delta
        print("Суммарная ошибка E = ", error_history[e])    
        e = e + 1 
        print("Вектор весов W = ", delta)
        print("Выходной вектор = ", y_net)
        
    return error_history

def test_full_set(w, X, targets, type):
    
    for i in range(len(X)):
        net = w[0] + w[1]*X[i][0] + w[2]*X[i][1] + w[3]*X[i][2] + w[4]*X[i][3]
        if type == '2':
                net = sigma(net)
        y = activate(net, type)
        if y != targets[i]:
            return False
    return True

def train_on_subset(subset_X, subset_targets, X, targets, type, max_epochs=50,verbose=True):
    
    w = np.zeros(5)
    epoch_errors = []
    epoch_outputs = []
    
    for epoch in range(max_epochs):
        epoch_error = 0
        epoch_output = []
        
        # Обучение на подмножестве
        for i in range(len(subset_X)):
            net = w[0] + w[1]*subset_X[i][0] + w[2]*subset_X[i][1] + w[3]*subset_X[i][2] + w[4]*subset_X[i][3]
            y = activate(net, type)
            error = subset_targets[i] - y
            
            if error != 0:
                if type == '1':
                    w[0] += 0.3 * error
                else:
                    w[0] += 0.3 * error*dsigma(net)
                    
                for j in range(1, 5):
                    if type == '1':
                            w[j] += 0.3 * error * subset_X[i][j-1]
                    else:
                            w[j] += 0.3 * error * subset_X[i][j-1]*dsigma(net)
        
        all_outputs = []
        for i in range(len(X)):
            
            net = w[0] + w[1]*X[i][0] + w[2]*X[i][1] + w[3]*X[i][2] + w[4]*X[i][3]
            y = activate(net, type)
            all_outputs.append(y)
            if y != targets[i]:
                epoch_error += 1
        
        epoch_errors.append(epoch_error)
        epoch_outputs.append(all_outputs)
        
        if epoch_error == 0:
            print(f"бучение завершено на эпохе {epoch}")
            return True, w, epoch + 1, epoch_errors, epoch_outputs
    
    return False, None, max_epochs, epoch_errors, epoch_outputs

def find_minimal_set(X, targets,type):
    indices = list(range(len(X)))
    
    # Идем от размера 1 до 16, перебирая все комбинации
    for size in range(1, len(X) + 1):
        
        for combo in combinations(indices, size):
            subset_X = [X[i] for i in combo]
            subset_targets = [targets[i] for i in combo]
            
            # Пытаемся обучиться на текущей комбинации
            success, w_final, epochs, epoch_errors, epoch_outputs = train_on_subset(
                subset_X, subset_targets, X, targets, type, max_epochs=50, verbose=True
            )
            
            if success:
                print(f"Размер набора: {size}")
                print(f"Индексы векторов (0-15): {combo}")
                print("\nВекторы в обучающем наборе:")
                print(f"{'i':>3} | {'x1 x2 x3 x4':^12} | t")
                print("-" * 30)
                for i in combo:
                    print(f"{i:3} | {X[i][0]} {X[i][1]} {X[i][2]} {X[i][3]} | {targets[i]}")
                print(f"Итоговые веса [w0, w1, w2, w3, w4]: {np.round(w_final, 4)}")
                
                return combo, w_final, epoch_errors, epoch_outputs
    
    return None, None, None, None

def save_output_to_file(filename="training_results.txt"):
    
    original_stdout = sys.stdout
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            sys.stdout = f
            
            print("="*80)
            print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ НЕЙРОНОВ")
            print("="*80)
            
            # Вывод таблицы истинности
            print("\n" + "="*80)
            print("ТАБЛИЦА ИСТИННОСТИ БУЛЕВОЙ ФУНКЦИИ")
            print("="*80)
            for i in range(16):
                x_str = f"{X[i][0]} {X[i][1]} {X[i][2]} {X[i][3]}"
                print(f"{i+1:2} | {x_str} | {targets[i]}")
            
            # Вывод обучения на полном наборе
            print("\n" + "="*80)
            print("ОБУЧЕНИЕ НА ПОЛНОМ НАБОРЕ ДАННЫХ")
            print("="*80)
            
            print("\n" + "-"*80)
            print("НЕЙРОН ТИПА 1 (ПОРОГОВАЯ ФУНКЦИЯ АКТИВАЦИИ)")
            print("-"*80)
            train_neuron('1')
            
            print("\n" + "-"*80)
            print("НЕЙРОН ТИПА 2 (СИГМОИДАЛЬНАЯ ФУНКЦИЯ АКТИВАЦИИ)")
            print("-"*80)
            train_neuron('2')
            
            # Поиск минимальных наборов
            print("\n" + "="*80)
            print("ПОИСК МИНИМАЛЬНЫХ ОБУЧАЮЩИХ НАБОРОВ")
            print("="*80)
            
            print("\n" + "-"*80)
            print("НЕЙРОН ТИПА 1 (ПОРОГОВАЯ ФУНКЦИЯ АКТИВАЦИИ)")
            print("-"*80)
            find_minimal_set(X, targets, '1')
            
            print("\n" + "-"*80)
            print("НЕЙРОН ТИПА 2 (СИГМОИДАЛЬНАЯ ФУНКЦИЯ АКТИВАЦИИ)")
            print("-"*80)
            find_minimal_set(X, targets, '2')
            
            print("\n" + "="*80)
            print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
            print("="*80)
            
    finally:
        sys.stdout = original_stdout
    
    print(f"Все результаты сохранены в файл: {filename}")

# Запускаем алгоритм поиска минимального набора
#train_neuron('1')
#train_neuron('2')

#find_minimal_set(X, targets, '1')

#find_minimal_set(X, targets, '2')

save_output_to_file("training_results.txt")

