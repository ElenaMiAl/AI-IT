import numpy as np
import matplotlib.pyplot as plt

def x_func(t):
    return np.sin(0.1 * t**3 - 0.2 * t**2 + t - 1)

a = 0        
b = 1          
N = 20         

t_pred_start = b
t_pred_end = 2*b - a 

p = 6          
eta = 1      
epochs = 4000  

print(f"Интервал обучения: t ∈ [{a}, {b}], N = {N} точек")
print(f"Интервал прогноза: t ∈ ({t_pred_start}, {t_pred_end}], N = {N} точек")
print(f"Размер окна p = {p}, норма обучения η = {eta}, эпох = {epochs}")


t_train = np.linspace(a, b, N)
x_train = x_func(t_train)

t_pred = np.linspace(t_pred_start, t_pred_end, N)
x_true_pred = x_func(t_pred)

print("\nИсходные данные")
for i in range(5):
    print(f"    t[{i}] = {t_train[i]:.4f}, x[{i}] = {x_train[i]:.6f}")

X_train = [] 
Y_train = []  

for i in range(p, len(x_train)):
    X_train.append(x_train[i-p:i])
    Y_train.append(x_train[i])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print(f"\nОбучающая выборка (скользящее окно p={p}):")
print(f"    Количество образцов: {len(X_train)}")
print(f"    Размер входа: {X_train.shape[1]}")
w = np.zeros(p)
errors = [] 

for epoch in range(epochs):
    epoch_error = 0
    
    for i in range(len(X_train)):
    
        y_pred = np.dot(w, X_train[i])
        delta = Y_train[i] - y_pred
        epoch_error += delta ** 2
        w += eta * delta * X_train[i]
    
    errors.append(epoch_error)
    
    if epoch % 500 == 0:
        print(f"Эпоха {epoch:4d}: ошибка = {epoch_error:.6f}")

print(f"Обучение завершено!")
print(f"Финальные веса: w = {[round(v, 4) for v in w]}")

print("\nПрогнозирование")

window = list(x_train[-p:])
predictions = []

for i in range(N):
    pred = np.dot(w, window)
    predictions.append(pred)
    # Сдвигаем окно
    window = window[1:] + [pred]

print(f"Первые 5 прогнозов: {[round(v, 4) for v in predictions[:5]]}")
print(f"Последние 5 прогнозов: {[round(v, 4) for v in predictions[-5:]]}")

train_error = errors[-1]

test_error = sum((predictions[i] - x_true_pred[i]) ** 2 for i in range(N))

print(f"\nОшибки:")
print(f"Суммарная квадратичная ошибка на обучении: E = {train_error:.6f}")
print(f"Суммарная квадратичная ошибка на прогнозе: ε = {test_error:.6f}")

print("\nПостроение графиков")

plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
t_continuous = np.linspace(a, b, 200)
x_continuous = x_func(t_continuous)
plt.plot(t_continuous, x_continuous, 'b-', linewidth=2, label='X(t)')
plt.plot(t_train, x_train, 'ro', markersize=6, label='x (обучающие точки)')

plt.xlabel('t')
plt.ylabel('X(t)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(t_train, x_train, 'bo', markersize=6, label='Исходные данные')
plt.plot(t_pred, x_true_pred, 'g-', linewidth=2, label='X(t) (истинная)')
plt.plot(t_pred, predictions, 'r--', linewidth=2, markersize=4, label='Прогноз НС')
plt.axvline(x=b, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label='Граница прогноза')

plt.xlabel('t')
plt.ylabel('X(t)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
