import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def boolean_function(x1, x2, x3, x4):
    return 1 if (not (x2 and x3)) or ((not x1) and x4) else 0

n = 4
X, targets = [], []

print("Таблица истинности:\n")
header = " i | " + " ".join([f"x{j+1}" for j in range(n)]) + " | F"
print(header)
print("-" * len(header))

for i in range(2**n):
    x = [(i >> j) & 1 for j in range(n-1, -1, -1)]
    t = boolean_function(*x)
    X.append(x)
    targets.append(t)
    x_str = ' '.join([str(v) for v in x])
    print(f"{i:2} | {x_str} | {t}")


def activate(net, type='1'):
    if type == '1':
        return 1.0 if net >= 0 else 0.0
    else:
        return 1.0 / (1.0 + np.exp(-net))

def derivative(net, out, type='1'):
    if type == '1':
        return 1.0
    else:
        return out * (1 - out)

def train_neuron(X, targets, eta=0.3, max_epochs=50, act_type='1', verbose=True, file=None):
    w = np.zeros(5)  
    loss_history = []
    weights_history = [w.copy()]
    outputs_history = []
    
    if verbose and file:
        file.write(f"\n--- Обучение с {act_type} ФА, η={eta} ---\n")
        file.write(f"{'Эпоха':>6} | {'Веса w0..w4':^45} | {'Выход (первые 8)':^30} | {'Ошибка':>8}\n")
        file.write("-" * 95 + "\n")
    
    for epoch in range(max_epochs):
        epoch_error = 0
        epoch_outputs = []
        
        for x, t in zip(X, targets):
            x_bias = [1.0] + x
            net = sum(w[i] * x_bias[i] for i in range(5))
            out_cont = activate(net, act_type)
            output = 1.0 if out_cont >= 0.5 else 0.0
            epoch_outputs.append(output)
            
            delta = t - output
            epoch_error += abs(delta)
            
            if abs(delta) > 1e-6:
                df = derivative(net, out_cont, act_type)
                for i in range(5):
                    w[i] += eta * delta * df * x_bias[i]
        
        loss_history.append(epoch_error)
        weights_history.append(w.copy())
        outputs_history.append(epoch_outputs)
        
        if verbose and file:
            w_str = f"[{w[0]:6.3f}, {w[1]:6.3f}, {w[2]:6.3f}, {w[3]:6.3f}, {w[4]:6.3f}]"
            out_str = ''.join(str(int(o)) for o in epoch_outputs[:16]) 
            file.write(f"{epoch:6d} | {w_str:45} | {out_str:30} | {epoch_error:8.2f}\n")
        
        if epoch_error == 0:
            if verbose and file:
                file.write("-" * 95 + "\n")
                file.write(f"Обучение завершено на эпохе {epoch+1}\n")
            break
    
    return w, loss_history, weights_history, outputs_history

def find_minimal_set(X, targets, eta=0.3, max_epochs=50, act_type='1', file=None):
    if file:
        file.write(f"Поиск минимального набора для {act_type} ФА...\n")
    
    for size in range(1, 9):
        for indices in combinations(range(16), size):
            X_subset = [X[i] for i in indices]
            t_subset = [targets[i] for i in indices]
            
            # Проверяем наличие обоих классов
            if 0 in t_subset and 1 in t_subset:
                w, loss, _, _ = train_neuron(X_subset, t_subset, eta, max_epochs, 
                                            act_type=act_type, verbose=False, file=None)
                
                if loss[-1] == 0:
                    if file:
                        file.write(f"    Найден набор из {size} векторов!\n")
                        file.write(f"    Индексы: {indices}\n")
                        file.write(f"    Векторы (x1,x2,x3,x4) -> F:\n")
                        for idx in indices:
                            file.write(f"      {X[idx]} -> {targets[idx]}\n")
                        file.write(f"    Эпох: {len(loss)}\n")
                        file.write(f"    Веса: [{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}, {w[3]:.3f}, {w[4]:.3f}]\n")
                    return indices, w, loss
    
    if file:
        file.write(f"Минимальный набор не найден\n")
    return None, None, None


eta = 0.3
max_epochs = 50

with open('lab1_results.txt', 'w', encoding='utf-8') as f:
    f.write("ЛАБОРАТОРНАЯ РАБОТА №1: ИССЛЕДОВАНИЕ ОДНОСЛОЙНЫХ НС\n")
    f.write("\nТаблица истинности БФ:\n")
    header = " i | " + " ".join([f"x{j+1}" for j in range(n)]) + " | F\n"
    f.write(header)
    f.write("-" * len(header) + "\n")
    for i, (x, t) in enumerate(zip(X, targets)):
        x_str = ' '.join([str(v) for v in x])
        f.write(f"{i:2} | {x_str} | {t}\n")
    
    f.write("\nОБУЧЕНИЕ С ПОРОГОВОЙ ФУНКЦИЕЙ АКТИВАЦИИ\n")
    w_step, loss_step, weights_step, outputs_step = train_neuron(
        X, targets, eta, max_epochs, act_type='1', verbose=True, file=f
    )
    
    f.write("\nОБУЧЕНИЕ С СИГМОИДАЛЬНОЙ ФУНКЦИЕЙ АКТИВАЦИИ\n")
    w_sig, loss_sig, weights_sig, outputs_sig = train_neuron(
        X, targets, eta, max_epochs, act_type='2', verbose=True, file=f
    )
    
    
    f.write("\nПОИСК МИНИМАЛЬНОГО ОБУЧАЮЩЕГО НАБОРА\n")
    min_step = find_minimal_set(X, targets, eta, max_epochs, act_type='1', file=f)
    min_sig = find_minimal_set(X, targets, eta, max_epochs, act_type='2', file=f)
    
    
    f.write("\nИТОГОВЫЕ РЕЗУЛЬТАТЫ\n")
    
    f.write("\nПороговая функция активации:\n")
    f.write(f"  Эпох обучения: {len(loss_step)}\n")
    f.write(f"  Финальные веса: w0 = {w_step[0]:.3f}, w1 = {w_step[1]:.3f}, w2 = {w_step[2]:.3f}, "
          f"w3 = {w_step[3]:.3f}, w4 = {w_step[4]:.3f}\n")
    
    f.write("\n Сигмоидальная функция активации:\n")
    f.write(f"  Эпох обучения: {len(loss_sig)}\n")
    f.write(f"  Финальные веса: w0 = {w_sig[0]:.3f}, w1 = {w_sig[1]:.3f}, w2 = {w_sig[2]:.3f}, "
          f"w3 = {w_sig[3]:.3f}, w4 = {w_sig[4]:.3f}\n")
    
    if min_step[0]:
        f.write("\n Минимальный набор (пороговая ФА):\n")
        f.write(f"  Индексы векторов: {min_step[0]}\n")
        f.write(f"  Размер набора: {len(min_step[0])}\n")
        f.write(f"  Эпох обучения: {len(min_step[2])}\n")
        f.write(f"  Финальные веса: [{min_step[1][0]:.3f}, {min_step[1][1]:.3f}, "
              f"{min_step[1][2]:.3f}, {min_step[1][3]:.3f}, {min_step[1][4]:.3f}]\n")
    
    if min_sig[0]:
        f.write("\n Минимальный набор (сигмоидальная ФА):\n")
        f.write(f"  Индексы векторов: {min_sig[0]}\n")
        f.write(f"  Размер набора: {len(min_sig[0])}\n")
        f.write(f"  Эпох обучения: {len(min_sig[2])}\n")
        f.write(f"  Финальные веса: [{min_sig[1][0]:.3f}, {min_sig[1][1]:.3f}, "
              f"{min_sig[1][2]:.3f}, {min_sig[1][3]:.3f}, {min_sig[1][4]:.3f}]\n")
    
print("\nРезультаты сохранены в файл 'lab1_results.txt'")

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(range(1, len(loss_step)+1), loss_step, 'b-o', linewidth=2, markersize=6, markerfacecolor='white')
plt.title(f'Пороговая функция активации\n(16 векторов, {len(loss_step)} эпох)')
plt.xlabel('Эпоха обучения k')
plt.ylabel('Суммарная ошибка E(k)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(0, len(loss_step)+1, max(1, len(loss_step)//5)))
plt.ylim(-0.5, 16.5)
plt.xlim(0, len(loss_step))

plt.subplot(2, 2, 2)
plt.plot(range(1, len(loss_sig)+1), loss_sig, 'r-o', linewidth=2, markersize=6, markerfacecolor='white')
plt.title(f'Сигмоидальная функция активации\n(16 векторов, {len(loss_sig)} эпох)')
plt.xlabel('Эпоха обучения k')
plt.ylabel('Суммарная ошибка E(k)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(0, len(loss_sig)+1, max(1, len(loss_sig)//5)))
plt.ylim(-0.5, 16.5)
plt.xlim(0, len(loss_sig))

if min_step[0]:
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(min_step[2])+1), min_step[2], 'g-o', linewidth=2, markersize=8, markerfacecolor='white')
    plt.title(f'Минимальный набор (пороговая ФА)\n{len(min_step[0])} векторов, {len(min_step[2])} эпох')
    plt.xlabel('Эпоха обучения k')
    plt.ylabel('Суммарная ошибка E(k)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(0, len(min_step[2])+1, max(1, len(min_step[2])//3)))
    plt.ylim(-0.5, len(min_step[0]) + 0.5)
    plt.xlim(0, len(min_step[2]))

if min_sig[0]:
    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(min_sig[2])+1), min_sig[2], 'm-o', linewidth=2, markersize=8, markerfacecolor='white')
    plt.title(f'Минимальный набор (сигмоидальная ФА)\n{len(min_sig[0])} векторов, {len(min_sig[2])} эпох')
    plt.xlabel('Эпоха обучения k')
    plt.ylabel('Суммарная ошибка E(k)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(0, len(min_sig[2])+1, max(1, len(min_sig[2])//3)))
    plt.ylim(-0.5, len(min_sig[0]) + 0.5)
    plt.xlim(0, len(min_sig[2]))

plt.tight_layout()
plt.savefig('lab1_graphs.png', dpi=150, bbox_inches='tight')
plt.show()

print("Графики сохранены в файл 'lab1_graphs.png'")
