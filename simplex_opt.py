import math
import numpy as np
import argparse
import sys
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.axisbelow'] = True
from matplotlib import cm
import seaborn as sns

sns.set_style('white')

parser = argparse.ArgumentParser(description=u'SimpleOpt Method')
args = parser.parse_args()

num_divs = 50
x_min = -5
x_max = 5
y_min = -5
y_max = 5
scope_x = [float(x_min), float(x_max)]
scope_y = [float(y_min), float(y_max)]

def get_symmetry_point(points, index):
    A = points[index]
    B = points[(index + 1) % 3]
    C = points[(index + 2) % 3]

    D = B + C - A
    return D


def get_symmetry_triangle(points, index):
    D = get_symmetry_point(points, index)
    new_points = np.copy(points)
    new_points[index] = D

    return new_points

def check_range_valid(x_min, x_max, y_min, y_max, x, y):
    if (x >= x_min and x <= x_max and y >= y_min and y <= y_max):
        return True

    return False


def is_triangle(points):
    if(len(points) != 3):
        print("len:", len(points))
        return False

    for i in range(3):
        A = points[i]
        B = points[(i + 1) % 3]
        C = points[(i + 2) % 3]

        AB = np.linalg.norm(A - B)
        AC = np.linalg.norm(A - C)
        BC = np.linalg.norm(B - C)

#         print("A:", A, "B:", B, "C:", C, "AB:", AB, "AC:", AC, "BC:", BC)

        if (AB + AC <= BC):
            return False

    return True

def func_sin(points):
    return np.sin(np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1])))

def func_square(points):
    return -(np.square(points[:, 0]) + np.square(points[:, 1]))

def func_exp(points):
    n_points = np.shape(points)[0]
    values = np.mat(np.zeros((n_points, 1)))
    for i in range(n_points):
        x = points[i, 0]
        y = points[i, 1]
        value  =  (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)
        values[i, 0] = value

    return values


def simplex_opt(x_min, x_max, y_min, y_max, x, y, a, obj_func, tol):
    triangle_list = []
    value_list = []

    error = np.inf
    while (error >= tol):
        p = a * (np.sqrt(3) + 1) / (2 * np.sqrt(2))
        q = a * (np.sqrt(3) - 1) / (2 * np.sqrt(2))
        triangle = np.mat([[x, y], [x + p, y + q], [x + q, y + p]])

        is_valid_triangle = is_triangle(triangle)
        if (is_valid_triangle == False):
            print("invalid triangle:", triangle)
            return

        optimal_point = np.mat([-np.inf, -np.inf])
        optimal_idx = -1;
        symmetry_point = np.mat([-np.inf + 1, -np.inf + 1])
        worst_point = np.mat([-np.inf, -np.inf])

        optimal_repeat_times = 0
        iter = 0;
        while (optimal_repeat_times < 3):
            iter += 1
            triangle_list.append(np.mat(np.copy(triangle)))
            fxy = obj_func(triangle)
            value_list.append(np.mat(np.copy(fxy).reshape(-1, 1)))
            min_idx = np.argmin(fxy)
            current_worst_point = triangle[min_idx]
            max_idx = np.argmax(fxy)
            current_optimal_point = np.mat(triangle[max_idx, :])
            worst_point = np.mat(triangle[min_idx, :])
            if ((current_optimal_point == optimal_point).all() == True):
                optimal_repeat_times += 1
            else:
                optimal_repeat_times = 1

            optimal_point = current_optimal_point
            optimal_idx = max_idx

            second_worst = 0;
            for i in range(3):
                if (i != min_idx and i != max_idx):
                    second_worst = i
                    break

            if ((symmetry_point == current_worst_point).all() == True):
                triangle = get_symmetry_triangle(triangle, second_worst)
                symmetry_point = triangle[second_worst]
            else:
                triangle = get_symmetry_triangle(triangle, min_idx)
                symmetry_point = triangle[min_idx]
                is_valid = check_range_valid(x_min, x_max, y_min, y_max, symmetry_point[0],
                                             symmetry_point[1])
                if (is_valid == False):
                    triangle = get_symmetry_triangle(triangle, min_idx)
                    triangle = get_symmetry_triangle(triangle, second_worst)
                    symmetry_point = triangle[second_worst]

        optimal_value = obj_func(optimal_point)
        worst_value = obj_func(worst_point)
        error = np.abs((optimal_value - worst_value) / optimal_value)
        print("optimal_point:", optimal_point, "optimal value:", optimal_value, "worst_value:", worst_value, "error:",
              error)
        optimal_repeat_times = 0

        if error >= tol:
            x = optimal_point[0, 0]
            y = optimal_point[0, 1]
            a = a / 2.0

    triangle_and_value_list = []
    for i in range(len(triangle_list)):
        triangle_and_value = np.copy(np.hstack((triangle_list[i], value_list[i])))
        triangle_and_value_list.append(triangle_and_value)

    optimal = np.copy(np.hstack((optimal_point, optimal_value)))
    return triangle_and_value_list, optimal


def func(x, y):
    return -(np.square(x) + np.square(y))
    # return np.sin(np.sqrt(np.square(x) + np.square(y)))
    #return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)



def snapshot2D(ax):
    a = np.linspace(scope_x[0] - 0.2, scope_x[1] + 0.2, num_divs)
    b = np.linspace(scope_y[0] - 0.2, scope_y[1] + 0.2, num_divs)
    x, y = np.meshgrid(a, b)
    # z=func(x,y)
    # ax.contour(x, y,z, levels=np.logspace(-3,3,25), cmap='jet')
    ax.contourf(x, y, func(x, y), 50, cmap=cm.hot, alpha=0.7)
    CS = ax.contour(x, y, func(x, y), 15, colors='k')
    ax.clabel(CS, inline=True, fontsize=13)
    ax.set_xlabel(r'x', fontdict={'fontsize': 18, 'fontweight': 'medium'})
    ax.set_ylabel(r'y', fontdict={'fontsize': 18, 'fontweight': 'medium'})

    ax.set_title(r'Simple_opt $f(x,y)$', fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)

    # ax.plot(final_x, final_y, marker='*', c='r', markersize=15)
    # ax.scatter(final_x, final_y, marker='*', c='r', s=50, label='Final Result')

    return ax



def snapshot3D(ax):
    a = np.linspace(scope_x[0] - 0.2, scope_x[1] + 0.2, num_divs)
    b = np.linspace(scope_y[0] - 0.2, scope_y[1] + 0.2, num_divs)
    X, Y = np.meshgrid(a, b)
    Z = func(X, Y)

    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, edgecolor='none', cmap='jet')
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, cmap='jet')
    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.5, cmap='jet', alpha=0.2)
    cset = ax.contourf(X, Y, Z, zdir='x', offset=scope_x[0] - 0.3, cmap='jet', alpha=0.2)
    cset = ax.contourf(X, Y, Z, zdir='y', offset=scope_y[1] + 0.3, cmap='jet', alpha=0.2)

    # CS = ax.contour(x, y, func(x, y), colors='k')
    # ax.clabel(CS, inline=True, fontsize=13)
    ax.set_xlabel(r'X', fontdict={'fontsize': 18, 'fontweight': 'medium'})
    ax.set_ylabel(r'Y', fontdict={'fontsize': 18, 'fontweight': 'medium'})
    ax.set_ylabel(r'Z', fontdict={'fontsize': 18, 'fontweight': 'medium'})

    ax.set_title(r'Simple opt $f(x,y)$ 3D', fontdict={'fontsize': 15, 'fontweight': 'medium'})
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)

    # ax.plot([final_x], [final_y], [final_z], marker='*', c='r', markersize=20, label='Final result')
    # ax.scatter([final_x], [final_y], [final_z], marker='*', c='r', s=80, label='Final result')

    # ax.plot(track_x[0], track_y[0], c='b', marker='o')

    return ax




def init3D():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])

    # point1.set_data([], [])
    # point1.set_3d_properties([])

    # point2.set_data([], [])
    # point2.set_3d_properties([])
    #
    # point3.set_data([], [])
    # point3.set_3d_properties([])

    display_value.set_text('')

    return line, point, display_value
    # return line, point, point1, point2, point3, display_value


def animate3D(i):
    # ti = 5
    a_tri = triangle_and_value_list[i]

    line.set_data(a_tri[:, 0], a_tri[:, 1])
    line.set_3d_properties(a_tri[:,2])

    point.set_data(a_tri[:, 0], a_tri[:, 1])
    point.set_3d_properties(a_tri[:,2])

    # point1.set_data(track_x[i], track_y[i])
    # point1.set_3d_properties([-0.5] * len(track_x[i]))

    # line.set_data(track_x[i], track_y[i])
    # line.set_3d_properties(track_z[i])
    #
    # point.set_data(track_x[i], track_y[i])
    # point.set_3d_properties(track_z[i])
    #
    # point1.set_data(track_x[i], track_y[i])
    # point1.set_3d_properties([-0.5] * len(track_x[i]))

    # point2.set_data(scope_x[0]* len(track_x[i]), track_y[i]) # a bug here
    # point2.set_3d_properties(track_z[i])
    # point2.set_data([scope_x[0] - 0.3] * len(track_x[i]), track_y[i])
    # point2.set_3d_properties(track_z[i])
    #
    # point3.set_data(track_x[i], [scope_y[1] + 0.3] * len(track_x[i]))
    # point3.set_3d_properties(track_z[i])

    display_value.set_text('Iteration: ' + str(i + 1) +
                           '\n'
        #                    r'$x(\alpha_1)= $' '{0:.4f}'.format(track_x[i][1]) + r' $x(\alpha_2)= $' '{0:.4f}'.format(
        # track_x[i][2]) +
        #                    '\n'
        #                    r'$y(\alpha_1)= $' '{0:.4f}'.format(track_y[i][1]) + r' $y(\alpha_2)= $' '{0:.4f}'.format(
        # track_y[i][2]) +
        #                    '\n'
        #                    r'$z(\alpha_1)= $' '{0:.4f}'.format(track_z[i][1]) + r' $z(\alpha_2)= $' '{0:.4f}'.format(
        # track_z[i][2])
                           )

    return line, point, display_value
    # return line, point, point1, point2, point3, display_value


if __name__ == '__main__':
    print("----test func_sin---")
    triangle_and_value_list, optimal = simplex_opt(-5, 5, -5, 5, 3, 2, 1.0, func_sin, 0.01)
    print(triangle_and_value_list, optimal)

    print("----test func_square---")
    triangle_and_value_list, optimal = simplex_opt(-5, 5, -5, 5, 3, 2, 1.0, func_square, 0.5)
    print(triangle_and_value_list, optimal)

    print("----test func_exp---")
    triangle_and_value_list, optimal = simplex_opt(-5, 5, -5, 5, 3, 2, 1.0, func_exp, 0.01)
    # print(triangle_and_value_list, optimal)

    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111)
    # ax=snapshot2D(ax)
    # plt.show()


    fig1 = plt.figure(figsize=(8, 8))
    ax2 = Axes3D(fig1)
    ax2 = snapshot3D(ax2)
    # ti = 5
    # a_tri=triangle_and_value_list[ti]
    #ax2.scatter([a_tri[:, 0]], [a_tri[:, 1]], [a_tri[:, 2]],  marker='*', c='r', s=80, label='Triangle')
    # ax2.scatter([a_tri[:, 0]], [a_tri[:, 1]], [a_tri[:, 2]], marker='o', c='r', s=80, label='Triangle')

    line, = ax2.plot([], [], [], 'r', lw=1.5)
    point, = ax2.plot([], [], [], 'ro', markersize=10, label='Experiments')  # on the surface
    # point1, = ax2.plot([], [], [], 'ro', markersize=10, alpha=0.2)  # projection to z
    # point2, = ax2.plot([], [], [], 'ro', markersize=10, alpha=0.2)  # projection to x
    # point3, = ax2.plot([], [], [], 'ro', markersize=10, alpha=0.2)  # projection to y

    display_value = ax2.text(scope_x[1], scope_y[1], 0.5, '', transform=ax2.transAxes)
    ax2.grid(False)
    ax2.legend(loc=1)

    anim = animation.FuncAnimation(fig1, animate3D, init_func=init3D,
                                   frames=len(triangle_and_value_list),
                                   interval=1000,
                                   repeat_delay=80, blit=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=2, metadata=dict(artist='L'), bitrate=1800)
    anim.save('simpleopt-001.mp4', writer=writer)

    anim.save('simpleopt-001.mp4', writer='imagemagick')
    anim.save('simpleopt-001.gif', writer='imagemagick', fps=2)
    plt.savefig('simpleopt-001.pdf')
    plt.savefig('simpleopt-001.png')

    plt.show()
