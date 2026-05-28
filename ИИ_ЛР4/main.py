import numpy as np
from numpy import tanh
from matplotlib import pyplot as plt
from itertools import product, combinations

J = 13
INPUT = 5 
OUTPUT = 1

eps = 0.01
nj = 0.1

def f(xs):
    
    x1, x2, x3, x4 = xs[0], xs[1], xs[2], xs[3]
    
    
    term1 = (1 - x1) or (1 - x2) or (1 - x3)
    
    
    term2 = (1 - x2) or (1 - x3) or x4
    
    return 1 if (term1 and term2) else 0

def activate_tanh(net):
    return 1/2 * (tanh(net) + 1)

def find_minimal_training_set(INPUT=4, J=13, OUTPUT=1, nj=0.1, eps=0.01, max_epoch=200):
    all_data = list(product([0,1], repeat=INPUT))
    
    for subset_size in range(1, len(all_data)+1):
        print(f"Пробуем размер подмножества: {subset_size}")
        for subset in combinations(all_data, subset_size):
            WEIGHTS = [
                np.random.randn(INPUT+1, J) * np.sqrt(1/(INPUT+1)),
                np.random.randn(J, OUTPUT) * np.sqrt(1/J)
            ]
            
            train_data = [list(d) for d in subset]
            
            for epoch in range(max_epoch):
                np.random.shuffle(train_data)
                err = 0
                for d in train_data:
                    xs = np.array(d + [1], dtype=float)
                    true_res = f(xs)
                    
                    z1 = np.dot(xs, WEIGHTS[0])
                    res_1 = 0.5*(np.tanh(z1)+1)
                    
                    z2 = np.dot(res_1, WEIGHTS[1])
                    res_2 = 0.5*(np.tanh(z2)+1)
                    
                    delta_2 = (true_res - res_2) * 2 * res_2 * (1 - res_2)
                    delta_1 = (delta_2 @ WEIGHTS[1].T) * 2 * res_1 * (1 - res_1)
                    
                    WEIGHTS[1] += nj * delta_2 * res_1.reshape(-1,1)
                    WEIGHTS[0] += nj * xs.reshape(-1,1) @ delta_1.reshape(1,-1)
                    
                    err += (true_res - res_2)**2
                
                if err < eps/10:
                    break
            
            correct = 0
            for x in all_data:
                xs = np.array(list(x) + [1], dtype=float)
                res_1 = 0.5*(np.tanh(np.dot(xs, WEIGHTS[0]))+1)
                res_2 = 0.5*(np.tanh(np.dot(res_1, WEIGHTS[1]))+1)
                pred = int(res_2.item() >= 0.5)
                true_res = f(xs)
                if pred == true_res:
                    correct += 1
            
            if correct == len(all_data):
                print(f"Минимальный набор найден! Размер: {subset_size}")
                return subset, WEIGHTS
    
    return None, None

# Поиск минимального обучающего набора
minimal_set, best_weights = find_minimal_training_set()
print("Минимальный набор для обучения:", minimal_set)

if minimal_set is None:
    print("Не удалось найти минимальный набор. Используем все данные.")
    all_data = list(product([0,1], repeat=4))
    minimal_set = all_data

data = [list(x) for x in minimal_set]

# Сброс весов и обучение на минимальном наборе
WEIGHTS = [
    np.random.randn(INPUT, J) * np.sqrt(1/INPUT),
    np.random.randn(J, OUTPUT) * np.sqrt(1/(INPUT+J))
]

err = 1
epoch = 0
errs_epoch = []
int_errs_epoch = []
max_epochs = 500

while err >= eps and epoch < max_epochs:
    np.random.shuffle(data)
    err = 0
    epoch += 1
    int_err = 0
    for d in data:
        xs = np.array(d + [1], dtype=float)
        true_res = f(xs)
        
        z1 = np.dot(xs, WEIGHTS[0])
        res_1 = 0.5*(np.tanh(z1)+1)
        
        z2 = np.dot(res_1, WEIGHTS[1])
        res_2 = 0.5*(np.tanh(z2)+1)
        
        delta_2 = (true_res - res_2) * 2 * res_2 * (1 - res_2)
        delta_1 = (delta_2 @ WEIGHTS[1].T) * 2 * res_1 * (1 - res_1)
        
        WEIGHTS[1] += nj * delta_2 * res_1.reshape(-1,1)
        WEIGHTS[0] += nj * xs.reshape(-1,1) @ delta_1.reshape(1,-1)
        
        err += (true_res - res_2)**2
        if round(res_2[0]) != true_res:
            int_err += 1
    errs_epoch.append(err)
    int_errs_epoch.append(int_err)

print(f"Обучение завершено за {epoch} эпох")

# Графики ошибок
plt.figure(figsize=(8,6))

plt.subplot(2,1,1)
plt.plot(range(1, len(errs_epoch)+1), int_errs_epoch)
plt.xlabel("Эпохи")
plt.ylabel("Кол-во ошибок")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(range(1, len(errs_epoch)+1), errs_epoch)
plt.xlabel("Эпохи")
plt.ylabel("Суммарная квадратичная ошибка")
plt.grid(True)

plt.tight_layout()
plt.show()

# Проверка после обучения
print("\n" + "="*70)
print("ТАБЛИЦА ИСТИННОСТИ")
print("="*70)
print(" x1  x2  x3  x4 | true | pred | raw output")
print("-"*70)

correct = 0
total = 0

for x in product([0,1], repeat=4):
    xs = np.array(list(x) + [1], dtype=float)
    true_res = f(xs)
    
    z1 = np.dot(xs, WEIGHTS[0])
    a1 = activate_tanh(z1)
    
    z2 = np.dot(a1, WEIGHTS[1])
    a2 = activate_tanh(z2)
    
    pred = int(a2.item() >= 0.5)
    plus = "  ERR" if pred != true_res else ""
    print(f" {x[0]}   {x[1]}   {x[2]}   {x[3]}   |   {true_res}    |    {pred}    |   {a2[0]:.6f}{plus}")
    
    if pred == true_res:
        correct += 1
    total += 1

print("-"*70)
print(f"\nТочность: {correct}/{total} ({correct/total*100:.1f}%)")

# Вывод весов
print("\n" + "="*70)
print("СИНАПТИЧЕСКИЕ ВЕСА ПОСЛЕ ОБУЧЕНИЯ")
print("="*70)
print("\nВеса скрытого слоя (W1):")
print(WEIGHTS[0])
print("\nВеса выходного слоя (W2):")
print(WEIGHTS[1])
