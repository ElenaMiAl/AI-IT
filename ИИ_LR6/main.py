import numpy as np

INPUT_LAYER = 1
J_LAYER = 2          # 2 нейрона в скрытом слое
OUTPUT_LAYER = 1

nj = 0.5

WEIGHTS = [
    np.zeros((INPUT_LAYER, J_LAYER), dtype=float),
    np.zeros((J_LAYER, OUTPUT_LAYER), dtype=float)
]

b1 = np.zeros(J_LAYER)
b2 = np.zeros(OUTPUT_LAYER)

def net(xs, w):
    return np.dot(xs, w)

def f(net):
    return np.tanh(net)

def df(net):
    return 1 - np.tanh(net)**2

err = 1
eps = 1 * 10**-3
xs = [-3]           # Вход = -3 (второе число из x = (1, -3))
t = [-0.1]          # Цель = -0.1 (из 10t = -1)

count = 0

print(f"{'Номер эпохи':<15} {'Выходное значение y':<25} {'Суммарная ошибка':<20}")
print("-" * 65)

while err > eps:
    count += 1

    ne = net(xs, WEIGHTS[0]) + b1
    res_1 = f(ne)
    
    ne_1 = net(res_1, WEIGHTS[1]) + b2
    res_2 = f(ne_1)
    
    err_vec = t - res_2
    err = np.sum(err_vec ** 2)
    
    delta_2 = df(ne_1) * err_vec
    delta_1 = delta_2.dot(WEIGHTS[1].T) * df(ne)
    
    b2 += nj * delta_2
    b1 += nj * delta_1
    
    WEIGHTS[1] += nj * np.outer(res_1, delta_2)
    WEIGHTS[0] += nj * np.outer(xs, delta_1)
    
    print(f"{count:<15} {res_2[0]:.6f}{'':<18} E({count})={err:.6f}")

print("-" * 65)
print(f"\nРезультат: y = {res_2[0]:.6f}, целевое t = {t[0]}")
