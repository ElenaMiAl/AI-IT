import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# ------------------ Метрики расстояния ------------------
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def chebyshev_distance(p1, p2):
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# ------------------ Алгоритм k-средних ------------------
def kmeans(points, centers, metric_func):
    points = np.array(points)
    centers = np.array(centers, dtype=float)
    k = len(centers)
    prev_centers = centers.copy() + 1  # чтобы начать цикл
    
    step = 0
    max_steps = 100
    
    while step < max_steps and not np.allclose(centers, prev_centers, atol=1e-4):
        prev_centers = centers.copy()
        
        # Присваиваем точки к ближайшим центрам
        labels = []
        for p in points:
            dists = [metric_func(p, c) for c in centers]
            labels.append(np.argmin(dists))
        labels = np.array(labels)
        
        # Новые центры = среднее арифметическое точек в кластере
        new_centers = []
        for j in range(k):
            cluster_points = points[labels == j]
            if len(cluster_points) > 0:
                new_centers.append(np.mean(cluster_points, axis=0))
            else:
                new_centers.append(centers[j])
        centers = np.array(new_centers)
        step += 1
    
    # Финальная разметка
    labels = []
    for p in points:
        dists = [metric_func(p, c) for c in centers]
        labels.append(np.argmin(dists))
    
    return centers, np.array(labels)

# ------------------ GUI приложение ------------------
class KMeansApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ЛР №9 — Кластеризация методом k-средних")
        self.root.geometry("1100x750")
        
        # Данные
        self.points = []  # список точек (x, y)
        self.centers = []  # начальные центры
        self.metric = tk.StringVar(value="Евклида")
        
        self.create_widgets()
        
    def create_widgets(self):
        # Левая панель управления
        left_frame = ttk.LabelFrame(self.root, text="Управление", width=280)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5, ipadx=5)
        
        # Ввод точек
        ttk.Label(left_frame, text="Добавить точку (x, y):").pack(pady=2)
        point_frame = ttk.Frame(left_frame)
        point_frame.pack(pady=2)
        self.x_entry = ttk.Entry(point_frame, width=8)
        self.x_entry.pack(side=tk.LEFT)
        ttk.Label(point_frame, text=",").pack(side=tk.LEFT)
        self.y_entry = ttk.Entry(point_frame, width=8)
        self.y_entry.pack(side=tk.LEFT)
        ttk.Button(left_frame, text="+ Добавить точку", command=self.add_point).pack(pady=2)
        
        # Кнопка для добавления примера из методички
        ttk.Button(left_frame, text="📋 Загрузить пример из методички", 
                   command=self.load_example).pack(pady=5)
        
        # Ввод центров кластеров
        ttk.Label(left_frame, text="Добавить центр кластера (x, y):").pack(pady=(10,2))
        center_frame = ttk.Frame(left_frame)
        center_frame.pack(pady=2)
        self.cx_entry = ttk.Entry(center_frame, width=8)
        self.cx_entry.pack(side=tk.LEFT)
        ttk.Label(center_frame, text=",").pack(side=tk.LEFT)
        self.cy_entry = ttk.Entry(center_frame, width=8)
        self.cy_entry.pack(side=tk.LEFT)
        ttk.Button(left_frame, text="+ Добавить центр", command=self.add_center).pack(pady=2)
        
        # Выбор метрики
        ttk.Label(left_frame, text="Метрика расстояния:").pack(pady=(10,2))
        metric_frame = ttk.Frame(left_frame)
        metric_frame.pack()
        ttk.Radiobutton(metric_frame, text="Евклида", variable=self.metric, 
                       value="Евклида", command=self.update_plot).pack(anchor=tk.W)
        ttk.Radiobutton(metric_frame, text="Чебышева", variable=self.metric, 
                       value="Чебышева", command=self.update_plot).pack(anchor=tk.W)
        ttk.Radiobutton(metric_frame, text="Манхэттен", variable=self.metric, 
                       value="Манхэттен", command=self.update_plot).pack(anchor=tk.W)
        
        # Кнопки управления
        ttk.Button(left_frame, text="▶ Запустить кластеризацию", 
                   command=self.run_clustering).pack(pady=10)
        ttk.Button(left_frame, text="🗑 Очистить всё", command=self.clear_all).pack(pady=5)
        
        # Таблица расстояний
        ttk.Label(left_frame, text="Расстояния до центров:", font=('Arial', 9, 'bold')).pack(pady=(10,2))
        self.dist_text = tk.Text(left_frame, height=8, width=30, font=('Courier', 8))
        self.dist_text.pack(pady=2, fill=tk.BOTH, expand=True)
        
        # Список точек
        ttk.Label(left_frame, text="Точки:").pack(pady=(10,2))
        self.points_listbox = tk.Listbox(left_frame, height=6)
        self.points_listbox.pack(fill=tk.BOTH, expand=True, padx=5)
        
        # Список центров
        ttk.Label(left_frame, text="Центры кластеров:").pack(pady=(5,2))
        self.centers_listbox = tk.Listbox(left_frame, height=3)
        self.centers_listbox.pack(fill=tk.BOTH, expand=True, padx=5)
        
        # Правая часть — график
        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.update_plot()
    
    def load_example(self):
        """Загружает пример из методички (стр. 49)"""
        self.clear_all()
        # Точки из примера
        example_points = [(143, 213), (180, 220), (183, 249), 
                         (271, 253), (226, 253), (315, 275), (266, 297)]
        for p in example_points:
            self.points.append(p)
            self.points_listbox.insert(tk.END, f"({p[0]}, {p[1]})")
        
        # Центры из примера
        example_centers = [(159, 238), (270, 278)]
        for c in example_centers:
            self.centers.append(c)
            self.centers_listbox.insert(tk.END, f"({c[0]}, {c[1]})")
        
        self.update_plot()
        messagebox.showinfo("Загружено", "Загружен пример из методички\n(7 точек, 2 центра)")
    
    def add_point(self):
        try:
            x = float(self.x_entry.get())
            y = float(self.y_entry.get())
            self.points.append((x, y))
            self.points_listbox.insert(tk.END, f"({x}, {y})")
            self.x_entry.delete(0, tk.END)
            self.y_entry.delete(0, tk.END)
            self.update_plot()
        except ValueError:
            messagebox.showerror("Ошибка", "Введите числа для x и y")
    
    def add_center(self):
        try:
            x = float(self.cx_entry.get())
            y = float(self.cy_entry.get())
            self.centers.append((x, y))
            self.centers_listbox.insert(tk.END, f"({x}, {y})")
            self.cx_entry.delete(0, tk.END)
            self.cy_entry.delete(0, tk.END)
            self.update_plot()
        except ValueError:
            messagebox.showerror("Ошибка", "Введите числа для x и y")
    
    def clear_all(self):
        self.points = []
        self.centers = []
        self.points_listbox.delete(0, tk.END)
        self.centers_listbox.delete(0, tk.END)
        self.dist_text.delete(1.0, tk.END)
        self.update_plot()
    
    def calculate_distances(self):
        """Вычисляет и отображает расстояния от каждой точки до центров"""
        if len(self.points) == 0 or len(self.centers) == 0:
            return
        
        metric_name = self.metric.get()
        if metric_name == "Евклида":
            metric_func = euclidean_distance
        elif metric_name == "Чебышева":
            metric_func = chebyshev_distance
        else:
            metric_func = manhattan_distance
        
        self.dist_text.delete(1.0, tk.END)
        self.dist_text.insert(tk.END, f"Метрика: {metric_name}\n")
        self.dist_text.insert(tk.END, "-" * 35 + "\n")
        self.dist_text.insert(tk.END, f"{'Точка':^12} | {'Расст. до центров':^20}\n")
        self.dist_text.insert(tk.END, "-" * 35 + "\n")
        
        for i, p in enumerate(self.points):
            dists = []
            for c in self.centers:
                dists.append(metric_func(p, c))
            
            dist_str = "  ".join([f"C{j+1}:{d:6.2f}" for j, d in enumerate(dists)])
            # Определяем ближайший центр
            min_idx = np.argmin(dists)
            self.dist_text.insert(tk.END, f"p{i+1}:{p!s:8} | {dist_str}\n")
            self.dist_text.insert(tk.END, f"          → ближайший: C{min_idx+1}\n")
    
    def run_clustering(self):
        if len(self.points) == 0:
            messagebox.showwarning("Предупреждение", "Добавьте хотя бы одну точку")
            return
        if len(self.centers) < 2:
            messagebox.showwarning("Предупреждение", "Добавьте хотя бы 2 центра кластеров")
            return
        
        metric_name = self.metric.get()
        if metric_name == "Евклида":
            metric_func = euclidean_distance
        elif metric_name == "Чебышева":
            metric_func = chebyshev_distance
        else:
            metric_func = manhattan_distance
        
        # Сохраняем начальные центры для отображения
        initial_centers = self.centers.copy()
        
        # Запускаем кластеризацию
        final_centers, labels = kmeans(self.points, self.centers, metric_func)
        
        # Визуализация результата
        self.ax.clear()
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        points_arr = np.array(self.points)
        
        for i in range(len(final_centers)):
            cluster_points = points_arr[labels == i]
            if len(cluster_points) > 0:
                self.ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                                color=colors[i % len(colors)], label=f'Кластер {i+1}', s=60, alpha=0.7)
        
        # Финальные центры
        self.ax.scatter(final_centers[:, 0], final_centers[:, 1], 
                        color='black', marker='X', s=250, label='Финальные центры', zorder=5)
        
        # Начальные центры (для сравнения)
        initial_arr = np.array(initial_centers)
        self.ax.scatter(initial_arr[:, 0], initial_arr[:, 1], 
                        color='gray', marker='o', s=150, label='Начальные центры', alpha=0.5, zorder=4)
        
        self.ax.set_title(f"Кластеризация k-средних (метрика: {metric_name})")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.canvas.draw()
        
        # Вывод результата
        result_str = f"Кластеризация завершена!\n"
        result_str += f"Метрика: {metric_name}\n"
        result_str += f"Начальные центры: {initial_centers}\n"
        result_str += f"Финальные центры:\n"
        for i, c in enumerate(final_centers):
            result_str += f"  Кластер {i+1}: ({c[0]:.2f}, {c[1]:.2f})\n"
        
        # Подсчёт точек по кластерам
        for i in range(len(final_centers)):
            count = np.sum(labels == i)
            result_str += f"  Кластер {i+1}: {count} точек\n"
        
        messagebox.showinfo("Результат кластеризации", result_str)
    
    def update_plot(self):
        self.ax.clear()
        
        # Отображаем точки
        if self.points:
            pts = np.array(self.points)
            self.ax.scatter(pts[:, 0], pts[:, 1], color='blue', label='Точки', s=60, alpha=0.7)
            
            # Подписываем точки
            for i, p in enumerate(self.points):
                self.ax.annotate(f' {i+1}', (p[0], p[1]), fontsize=8, alpha=0.7)
        
        # Отображаем центры
        if self.centers:
            cents = np.array(self.centers)
            self.ax.scatter(cents[:, 0], cents[:, 1], color='red', marker='X', s=200, label='Центры', zorder=5)
            
            # Подписываем центры
            for i, c in enumerate(self.centers):
                self.ax.annotate(f' C{i+1}', (c[0], c[1]), fontsize=10, fontweight='bold', color='darkred')
        
        # Рисуем линии от точек к ближайшим центрам (для визуализации)
        if len(self.points) > 0 and len(self.centers) > 1:
            metric_name = self.metric.get()
            if metric_name == "Евклида":
                metric_func = euclidean_distance
            elif metric_name == "Чебышева":
                metric_func = chebyshev_distance
            else:
                metric_func = manhattan_distance
            
            # Определяем принадлежность точек к центрам
            for p in self.points:
                dists = [metric_func(p, c) for c in self.centers]
                nearest_idx = np.argmin(dists)
                nearest_center = self.centers[nearest_idx]
                # Рисуем пунктирную линию
                self.ax.plot([p[0], nearest_center[0]], [p[1], nearest_center[1]], 
                            'gray', linestyle=':', alpha=0.3, linewidth=0.8)
        
        self.ax.set_title("Исходные данные (пунктир — к ближайшему центру)")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.canvas.draw()
        
        # Обновляем таблицу расстояний
        self.calculate_distances()

# ------------------ Запуск ------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = KMeansApp(root)
    root.mainloop()
