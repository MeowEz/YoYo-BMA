import random as r
import math as m
import statistics as st
import time as t
import matplotlib.pyplot as plt

n = 100  # общее число особей
man_n = 49  # количество человек
yoyo_n = n - man_n  # количество йо-йо
dimen = 5  # мерность пространства
c_min = -100  # минимум пространства поиска
c_max = 100  # максимум пространства поиска
t_end = 1  # время окончания программы
sm_yoyo_n = 5  # количество маленьких йо-йо
lg_threshold = 1
step_size = 1
size_step = 0.05
sm_alpha = 0.00001
sm_threshold = 0.000000001
bl_omeg = 80
bl_gam = 1
bl_lamb = 4
times = (5, 15)
dimens = (5, 10)  # (2,)


def fit_func(x):  # функция sum of time
    return sum([(abs(x[i]) ** (i + 2)) for i in range(dimen)])


def generate_coor():
    a = []
    for i in range(dimen):
        a.append(r.uniform(c_min, c_max))
    return a


def dist_check(a, b, dist):  # проверка на расстояние
    c = m.sqrt(sum([((a[i] - b[i]) ** 2) for i in range(dimen)]))
    if c < dist:
        return 1
    else:
        return 0


def negpos():
    if r.random() < 0.5:
        return 1
    else:
        return -1


for var_num in range(10):
    print(var_num)
    for dimen in dimens:
        # TRID вставь сюда c_min = -(d**2) и c_max = d**2
        for t_end in times:
            var_num = 0
            plt.ylabel('Результат')
            plt.xlabel('Итерации')
            graph_name = '_'.join(['sphere', str(var_num + 1), str(dimen), str(t_end), '.png'])
            graph_x = []
            graph_y = []

            man = []  # список с мужчинами
            lg_yoyo = []  # список с длинными йо-йо

            for i in range(yoyo_n):  # генерация координат йо-йо
                lg_yoyo.append([])
                lg_yoyo[i].append(generate_coor())  # генерируем координаты
                lg_yoyo[i].append(fit_func(lg_yoyo[i][0]))  # вычисляем значение фитнес-функции

            j = 0
            for i in range(man_n):  # генерация координат мужчин
                man.append([])
                man[i].append(generate_coor())  # генерируем координаты
                man[i].append(fit_func(man[i][0]))  # вычисляем значение фитнес-функции
                man[i].append([])  # добавляем список с номерами йо-йо
                if j < yoyo_n:  # выдача йо-йо мужчинам
                    man[i][2].append(j)
                    j += 1
            man.sort(key=lambda i: i[1])  # сортировка мужчин по значениям фитнес функции
            best = list(man[0])

            while j < yoyo_n:  # раздаем оставшиеся йо йо
                man[man.index(r.choice(man))][2].append(j)
                j += 1

            gen = 0
            t_start = t.time()

            while (t.time() - t_start) < t_end * 60:
                for i in range(man_n):  # проход по мужчинам
                    for j in man[i][2]:  # проход по йо-йо выбранного мужика
                        while dist_check(man[i][0], lg_yoyo[j][0],
                                         lg_threshold) == 0:  # проверка расстояния между мужиком и йо-йо
                            if lg_yoyo[j][1] < man[i][1]:  # если йо-йо расположена в лучшем месте, чем мужик
                                man[i][0] = lg_yoyo[j][0]  # ставим мужика на место этого йо-йо
                                man[i][1] = lg_yoyo[j][1]
                            else:  # иначе
                                for k in range(dimen):
                                    dist = lg_yoyo[j][0][k] - man[i][0][k]  # приближаем йо-йо
                                    move_step = dist * r.random() * step_size
                                    lg_yoyo[j][0][k] -= move_step
                                lg_yoyo[j][1] = fit_func(lg_yoyo[j][0])
                        lg_yoyo[j][0] = generate_coor()  # генерируем новые значения для йо-йо
                        lg_yoyo[j][1] = fit_func(lg_yoyo[j][0])
                man.sort(key=lambda i: i[1])
                # ---------------------------------МАЛЕНЬКИЕ ЙО ЙО-----------------------------------------------------
                sm_yoyo = []
                if best != man[0]:
                    best = list(man[0])
                    sm_dimen = r.randint(0, dimen - 1)  # рандомный выбор мерностей для смещения маленьких йо-йо
                    for j in range(sm_yoyo_n):  # генерация маленьких йо-йо
                        sm_yoyo.append([])  # создание маленького йо-йо
                        sm_yoyo[j].append([])  # добавление пустого списка в первую ячейку
                        for k in range(dimen):
                            if k <= sm_dimen:  # смещение относительно выбраного количества мерностей
                                sm_yoyo[j][0].append(man[0][0][k] + (negpos() * sm_alpha * r.random()))
                            else:
                                sm_yoyo[j][0].append(man[0][0][k])
                        sm_yoyo[j].append(fit_func(sm_yoyo[j][0]))  # вычисление фитнес функции маленького йо йо
                    while sm_yoyo != []:  # пока еще есть маленькие йо йо
                        for j in range(len(sm_yoyo)):
                            if sm_yoyo[j][1] < man[0][1]:
                                man[0][0] = sm_yoyo[j][0]
                                man[0][1] = sm_yoyo[j][1]
                                sm_yoyo[j] = 0
                            else:
                                for i in range(dimen):
                                    dist = sm_yoyo[j][0][i] - man[0][0][i]
                                    move_step = dist * (r.random() + size_step)
                                    sm_yoyo[j][0][i] -= move_step
                                if dist_check(sm_yoyo[j][0], man[0][0], sm_threshold) == 1:
                                    sm_yoyo[j] = 0
                        sm_yoyo = [x for x in sm_yoyo if x != 0]
                # -----------------------------------------------СЛЕПОЙ ОПЕРАТОР---------------------------------------------
                bl_k = r.randint(0, int(dimen / 100 * bl_omeg))
                bl_k = 3
                bl_selected_dim = []
                bl_step = []
                for i in range(bl_k):
                    bl_selected_dim.append(r.randint(0, dimen - 1))
                    bl_step.append([])
                    bl_step[i].append([])
                    for j in range(dimen):
                        if j == bl_selected_dim[i]:
                            bl_step[i][0].append(
                                man[0][0][j] + (man[0][0][j] * r.uniform(-1, 1) * bl_gam) ** r.randint(1, bl_lamb))
                        else:
                            bl_step[i][0].append(man[0][0][j])
                    bl_step[i].append(fit_func(bl_step[i][0]))
                bl_step.sort(key=lambda i: i[1])
                if bl_step[0][1] < man[0][1]:
                    man[0][0] = bl_step[0][0]
                    man[0][1] = bl_step[0][1]
                # ----------------------------------------------МАЛЕНЬКИЕ ЙО ЙО----------------------------------------------
                sm_yoyo = []
                sm_dimen = r.randint(0, dimen - 1)  # рандомный выбор мерностей для смещения маленьких йо-йо
                for j in range(sm_yoyo_n):  # генерация маленьких йо-йо
                    sm_yoyo.append([])  # создание маленького йо-йо
                    sm_yoyo[j].append([])  # добавление пустого списка в первую ячейку
                    for k in range(dimen):
                        if k <= sm_dimen:  # смещение относительно выбраного количества мерностей
                            sm_yoyo[j][0].append(man[0][0][k] + (negpos() * sm_alpha * r.random()))
                        else:
                            sm_yoyo[j][0].append(man[0][0][k])
                    sm_yoyo[j].append(fit_func(sm_yoyo[j][0]))  # вычисление фитнес функции маленького йо йо
                while sm_yoyo != []:  # пока еще есть маленькие йо йо
                    for j in range(len(sm_yoyo)):
                        if sm_yoyo[j][1] < man[0][1]:
                            man[0][0] = sm_yoyo[j][0]
                            man[0][1] = sm_yoyo[j][1]
                            sm_yoyo[j] = 0
                        else:
                            for i in range(dimen):
                                dist = sm_yoyo[j][0][i] - man[0][0][i]
                                move_step = dist * (r.random() + size_step)
                                sm_yoyo[j][0][i] -= move_step
                            if dist_check(sm_yoyo[j][0], man[0][0], sm_threshold) == 1:
                                sm_yoyo[j] = 0
                    sm_yoyo = [x for x in sm_yoyo if x != 0]
                gen += 1
                if best != man[0]:
                    best = list(man[0])
                    graph_x.append(gen)
                    graph_y.append(best[1])
            print(dimen, t_end, best[1])
            plt.title(graph_name)
            plt.yscale('log')
            plt.plot(graph_x, graph_y, linewidth=2)
            plt.savefig(graph_name)
            plt.clf()