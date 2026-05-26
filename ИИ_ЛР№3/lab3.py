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
eta = 0.05     
epochs = 2000  

print(f"Интервал обучения: t ∈ [{a}, {b}], N = {N} точек")
print(f"Интервал прогноза: t ∈ ({t_pred_start}, {t_pred_end}], N = {N} точек")
print(f"Размер окна p = {p}, норма обучения η = {eta}, эпох = {epochs}")

t_train = np.linspace(a, b, N)
x_train = x_func(t_train)

t_pred = np.linspace(t_pred_start, t_pred_end, N)
x_true_pred = x_func(t_pred)

# Формирование обучающей выборки (скользящее окно)
X_train = [] 
Y_train = []  

for i in range(p, len(x_train)):
    X_train.append(x_train[i-p:i])
    Y_train.append(x_train[i])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Обучение
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

print(f"Финальные веса: w = {[round(v, 4) for v in w]}")

# Прогнозирование
window = list(x_train[-p:])
predictions = []

for i in range(N):
    pred = np.dot(w, window)
    predictions.append(pred)
    window = window[1:] + [pred]

# --- ГРАФИК КАК В МЕТОДИЧКЕ ---
plt.figure(figsize=(12, 6))

# Сплошная линия - истинная функция X(t) на всём интервале
t_full = np.linspace(a, t_pred_end, 500)
x_full = x_func(t_full)
plt.plot(t_full, x_full, 'b-', linewidth=1.5, label='Истинная X(t)', alpha=0.7)

# Точки - обучающие данные
plt.plot(t_train, x_train, 'bo', markersize=6, label='Обучающие данные')

# Прогноз (красная линия)
plt.plot(t_pred, predictions, 'r-', linewidth=2, label='Прогноз НС')

# Вертикальная разделительная линия
plt.axvline(x=b, color='black', linestyle='--', linewidth=1.5, label='Граница прогноза')

# Точки прогноза (опционально, можно убрать)
plt.plot(t_pred, predictions, 'ro', markersize=4)

# Оформление
plt.xlabel('t')
plt.ylabel('X(t)')
plt.title('Прогнозирование временного ряда с помощью линейной нейронной сети\n(метод скользящего окна)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.2)

# Можно настроить пределы осей
plt.xlim(a - 0.1, t_pred_end + 0.1)
# plt.ylim(-1.2, 1.2)  # раскомментировать, если нужно фиксировать по Y

plt.tight_layout()
plt.show()

# Вывод ошибок
train_error = errors[-1]
test_error = sum((predictions[i] - x_true_pred[i]) ** 2 for i in range(N))

print(f"\nОшибки:")
print(f"Суммарная квадратичная ошибка на обучении: E = {train_error:.6f}")
print(f"Суммарная квадратичная ошибка на прогнозе: ε = {test_error:.6f}")
