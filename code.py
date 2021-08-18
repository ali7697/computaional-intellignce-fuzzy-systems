from copy import deepcopy
import matplotlib.pyplot as plt
import csv
import pylab
import numpy as np
import random
import math


m = 1.75
# reading the inputs
file_location = "input file location here"
inputs = []
with open(file_location, 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        inputs.append(row)
length = len(inputs[0])
for i in range(len(inputs)):
    for j in range(length):
        inputs[i][j] = float(inputs[i][j])



def calculate_c(us, vs):
    entropy = 0
    for ii in range(len(inputs)):
        for jj in range(len(vs)):
            num = us[ii][jj]
            entropy += -(num * math.log(num))
    return entropy


def v_initializer(num_of_clusters):
    centers = []
    for q in range(num_of_clusters):
        tmp_c = []
        for w in range(length):
            tmp_c.append(float(random.random()))
        centers.append(tmp_c)
    return centers


def cal_norm(a, b):
    temp_sum = 0
    for i in range(length):
        temp_sum += (float(a[i]) - float(b[i])) ** 2
    result = math.sqrt(temp_sum)
    return result


def u_update(u, v1):
    for y in range(len(v1)):
        for k in range(len(inputs)):
            nominator = cal_norm(inputs[k], v1[y])
            total = 0
            for kk in range(len(v1)):
                total += (nominator / cal_norm(inputs[k], v1[kk])) ** (2 / (m - 1))
            u[k][y] = 1 / total


old_vs = []


def v_update(u, v):
    global old_vs
    old_vs = []
    old_vs = deepcopy(v)
    for i in range(len(v)):
        summ = 0
        nu = 0
        for k in range(len(inputs)):
            summ += u[k][i] ** m
        for k in range(len(inputs)):
            nu += (u[k][i] ** m) * np.array(inputs[k])
        v[i] = nu / summ


def is_done(vs_old, vs_new):
    distances = []
    for index in range(len(vs_old)):
        distances.append(cal_norm(vs_old[index], vs_new[index]))
    if max(distances) < 0.001:
        return True
    return False


def cal_cost(u, v):
    nu = 0
    for k in range(len(inputs)):
        for w in range(len(v)):
            nu += (u[k][w] ** m) * (cal_norm(inputs[k], v[w]) ** 2)
    return nu


us_for_c = []
ents = []
costs = []
# test different number of clusters and pick the best
for c in range(2, 10):
    us_for_c = [[0 for i in range(c)] for j in range(len(inputs))]
    vs = []
    vs = v_initializer(c)
    while True:
        u_update(us_for_c, vs)
        v_update(us_for_c, vs)
        if is_done(old_vs, vs):
            break
    ent = calculate_c(us_for_c, vs)
    ent = ent / math.log(c)
    ents.append(ent)
    costs.append(cal_cost(us_for_c, vs))
# print number of clusters and other useful information
c_number = ents.index(min(ents)) + 2
print("num of clusters:")
print(c_number)
print("entropies: ")
print(ents)
print("costs: ")
print(costs)
# do the computations for the best number of clusters
us = [[0 for i in range(c_number)] for j in range(len(inputs))]
vs = v_initializer(c_number)
while True:
    u_update(us, vs)
    v_update(us, vs)
    if is_done(old_vs, vs):
        break
clusters_of_the_points = []
for i in range(len(inputs)):
    clusters_of_the_points.append(us[i].index(max(us[i])))
# if data is two dimensional draw it
if (length == 2):
    colors = []
    cm = pylab.get_cmap('gist_rainbow')
    for i in range(c_number):
        colors.append(cm(1. * i / c_number))
    for v in range(len(inputs)):
        plt.scatter(inputs[v][0], inputs[v][1], color=colors[clusters_of_the_points[v]])
    for v in range(len(vs)):
        plt.scatter(vs[v][0], vs[v][1], color='black')
    plt.savefig("main.png")
    plt.show()


    def generate_random_points():
        # lets take n = 20
        n = 20
        output = []
        for z in range(n + 1):
            for x in range(n + 1):
                output.append([x / n, z / n])
        return output

    # the points that will be used for the second part of the project (emtiazi)
    border_points = generate_random_points()


    def find_close_point_indexes(point):
        num_close_points = 5
        distances = []
        indexes = []
        for index in range(len(inputs)):
            distances.append(cal_norm(inputs[index], point))
        dist2 = deepcopy(distances)
        distances.sort()
        tmp = distances[0:num_close_points]
        for k in range(len(tmp)):
            indexes.append(dist2.index(tmp[k]))
        return indexes


    def cal_us(points):
        caled_us = []
        for point in points:
            temp_u = []
            sum_denominator = 0
            sum_nominator = [0] * c_number
            indexes = find_close_point_indexes(point)
            for inp_index in indexes:
                val = cal_norm(point, inputs[inp_index]) ** (2 / (m - 1))
                sum_denominator += val
                sum_nominator[clusters_of_the_points[inp_index]] += val
            for k in range(c_number):
                temp_u.append(sum_nominator[k] / sum_denominator)
            caled_us.append(temp_u)
        return caled_us


    calculated_us = cal_us(border_points)
    cluster_of_border_points = []
    for i in range(len(border_points)):
        cluster_of_border_points.append(calculated_us[i].index(max(calculated_us[i])))
    for v1 in range(len(border_points)):
        plt.scatter(border_points[v1][0], border_points[v1][1], color=colors[cluster_of_border_points[v1]])
    plt.savefig("emtiazi.png")
    plt.show()
