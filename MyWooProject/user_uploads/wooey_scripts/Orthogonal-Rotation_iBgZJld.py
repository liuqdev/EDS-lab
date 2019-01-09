import argparse
import sys
import numpy as np
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

##############################
delta = 0.618
x_min = 0
x_max = 8
y_min = 0
y_max = 8
num = 30
num_divs = 50
tol = 0.0001

init_x = 1.0
init_y = 0.0

final_x = None
final_y = None
final_z = None

scope_x = [float(x_min), float(x_max)]
scope_y = [float(y_min), float(y_max)]
track_x = []
track_y = []

# function parameters, because origianal function design failure, there should be
# many parameters. My though is FFT, convert any function into f(x,y)=a0+a1*x^1+...+an*x^n
X_PARAM = [1., 0., 1., 0., 0, 0, 0, float('nan'), float('nan')]
Y_PARAM = [1., 0., 1., 0., 0, 0, 0, float('nan'), float('nan')]
CHOOSE = -1


#############################


def pkl(path, d):
    import pickle
    with open(path, 'wb') as fo:
        pickle.dump(d, fo)
        print('SUSS')


def unpkl(path):
    import pickle
    with open(path, 'rb') as fo:
        d = pickle.load(fo)
        print('SUSS')
    return d


def union_factor(init_x, init_y):
    init = (init_x, init_y)  # input()

    opt_x = init[0]
    iter = 1
    while True:
        print('\n----------------Iteration {0:0>3d}-------------------'.format(iter))
        print("--------------Keep x = {0:.4f} fixed-------------".format(round(opt_x, 4)))
        opt_y, f_opt_y = single_factor(['x', opt_x], scope_y)

        print("--------------Keep y = {0} fixed-------------".format(round(opt_y, 4)))
        opt_x, f_opt_x = single_factor(['y', opt_y], scope_x)

        if abs(f_opt_y - f_opt_x) < tol:
            # print("The Optimal indicator:%f" % (f_opt_x))
            # print("The Factor value:%f,%f" % (opt_x, opt_y))
            print()
            print('Find best indicator at (x={0:.4f}, y={1:.4f}), values={2:.4f}'.format(opt_x, opt_y, f_opt_x))
            break
        else:
            iter += 1

    return opt_x, opt_y, f_opt_x


def single_factor(axis, scope):
    fix = axis[1]
    bord_left = scope[0]
    bord_right = scope[1]
    alpha_0, alpha_1 = calc_alpha(bord_left, bord_right)

    termin = True
    itera = 1
    while termin:
        t = 'y' if axis[0] == 'x' else 'x'
        # print('Keep {0} fixed, searing on {1}'.format(t, axis[0]))
        f_0 = calc_func(axis[0], fix, alpha_0)
        f_1 = calc_func(axis[0], fix, alpha_1)

        # rngs=np.array(sorted([round(f_0, 4), round(f_1, 4), round(alpha_0, 4), round(alpha_1, 4)]))
        # print(type(rngs), rngs.dtype)

        # vals=func(rngs)

        # print('Experiments {0} on {1} = {2}, results={3} '.format(itera, t, rngs, vals))

        # if  axis[0] == 'x' :
        #     ty=rngs(np.argmax(vals)[0])
        #     print('The best result {0} found at x={1}, y={2}'.format(np.max(vals), fix, ty))
        # else:
        #     tx = rngs(np.argmax(vals)[0])
        #     print('The best result {0} found at x={1}, y={2}'.format(np.max(vals), tx, fix ))

        # print(f_1)
        # print(alpha_0)
        # print(alpha_1)
        if axis[0] == 'x':
            track_x.append([fix] * 4)
            track_y.append([bord_left, alpha_0, alpha_1, bord_right])
        elif axis[0] == 'y':
            track_y.append([fix] * 4)
            track_x.append([bord_left, alpha_0, alpha_1, bord_right])
        bord_left, bord_right, termin = bord_update([bord_left, bord_right], [alpha_0, f_0], [alpha_1, f_1], termin)
        alpha_0, alpha_1 = calc_alpha(bord_left, bord_right)
        itera = itera + 1
        opt = (alpha_0 + alpha_1) / 2.

        # rngs = np.array([bord[0], seg0[0], seg1[0], bord[1]])
        # rngs = np.array([bord_left, alpha_0, alpha_1, bord_right])
        rngs = np.array([round(bord_left, 4), round(alpha_0, 4), round(alpha_1, 4), round(bord_right, 4)])
        if t == 'x':
            vals = np.array([round(func(x, fix), 4) for x in rngs])
        else:
            vals = np.array([round(func(fix, y), 4) for y in rngs])

        print('Experiments {0:0>2d} on {1} = {2}, results={3} '.format(itera, t, rngs, vals))

        # rngs = np.array([bord_left, round(alpha_0, 4), round(alpha_1, 4), round(bord_right, 4)])

    f_opt = calc_func(axis[0], fix, opt)
    return opt, f_opt


def bord_update(bord, seg0, seg1, termin):
    bord_left = bord[0]
    bord_right = bord[1]
    # gap = abs(seg0[1] - seg1[1])
    # print(type(seg0[1]))
    # print(type(seg1[1]))
    gap = abs(seg0[1] - seg1[1])  #
    if gap > 0.001:
        # print('------------------------------a')
        if seg0[1] > seg1[1]:
            # print('------------------------------a1')
            bord_right = seg1[0]
            bord_left = bord[0]
        else:
            # print('------------------------------a2')
            bord_right = bord[1]
            bord_left = seg0[0]
    else:
        # print('------------------------------b')
        # print('gap={0} seg0={1} seg1={2}'.format(gap, seg0[0], seg1[1]))
        termin = False

    # rngs = np.array([bord[0], seg0[0], seg1[0], bord[1]])
    # print('')
    #

    # if t == 'x':
    #     vals = np.array([func(x, fix) for x in rngs])
    # else:
    #     vals = np.array([func(fix, y) for y in rngs])
    #
    # print('Experiments {0} on {1} = {2}, results={3} '.format(itera, t, rngs, vals))

    return bord_left, bord_right, termin


def calc_alpha(bord_left, bord_right):
    gap = bord_right - bord_left

    alpha_0 = bord_right - delta * gap
    alpha_1 = bord_left + delta * gap

    return alpha_0, alpha_1


def func(x, y, choose=CHOOSE):
    if choose == -1:
        return (x - 4) * np.exp(-(((x - 5) / 2.) ** 2 + ((y - 4.5) / 1.5) ** 2))  # just for testing, no parameters
    elif choose == 0:
        # print('x={0} y={1}'.format(x, y))
        # print(
        #     'f(x,y) = {0}*(x-{1})^{2}+{3}+ {4}*(y-{5})^{6}+{7} '.format(X_PARAM[0], X_PARAM[1], X_PARAM[2], X_PARAM[3],
        #                                                                  Y_PARAM[0], Y_PARAM[1], Y_PARAM[2],
        #                                                                  Y_PARAM[3]))
        z = X_PARAM[0] * (x - X_PARAM[1]) ** X_PARAM[2] + Y_PARAM[0] * (y - Y_PARAM[1]) ** Y_PARAM[2] + X_PARAM[3] + \
            X_PARAM[3]  # X_PARAMS 1 0 2; Y_PARAMS 1 0 2
        return z
    elif choose == 1:
        # print('x={0} y={1}'.format(x, y))
        # print('f(x,y) = sin(sqrt({0}*(x-{1})^{2})+{3}+ {4}*(y-{5})^{6}+{7} '.format(X_PARAM[0], X_PARAM[1], X_PARAM[2], X_PARAM[3], Y_PARAM[0], Y_PARAM[1], Y_PARAM[2], Y_PARAM[3]  ))
        return np.sin(
            np.sqrt(X_PARAM[0] * (x - X_PARAM[1]) ** X_PARAM[2] + Y_PARAM[0] * (y - Y_PARAM[1]) ** Y_PARAM[2])) + \
               X_PARAM[3] + X_PARAM[3]
    elif choose == -2:  # not recomanded
        z = (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
        # z1= X_PARAM[0]*(x-X_PARAM[1])**X_PARAM[2]+ X_PARAM[3] + X_PARAM[4]*x**4+X_PARAM[5]*x**5 +Y_PARAM[0]*(x-Y_PARAM[1])**Y_PARAM[2]+ Y_PARAM[3] + Y_PARAM[4]*x**4+Y_PARAM[5]*x**5
        # z1 = ( X_PARAM[0] * (x - X_PARAM[1]) ** X_PARAM[2] + X_PARAM[3] + X_PARAM[4] * x ** 4 + X_PARAM[5] * x ** 5 + Y_PARAM[
        #     0] * (x - Y_PARAM[1]) ** Y_PARAM[2] + Y_PARAM[3] + Y_PARAM[4] * x ** 4 + Y_PARAM[5] * x ** 5)*np.exp(-x**2-y**2)
        # too many parameters, undo it
        return z
    else:
        return -x ** 2 - y ** 2
        #  (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)


#
# def func(x, y, choose=CHOOSE):
#     if choose==-1:
#         return (x - 4) * np.exp(-(((x - 5) / 2.) ** 2 + ((y - 4.5) / 1.5) ** 2)) # just for testing, no parameters
#     elif choose==0:
#         z = -X_PARAM[0]*(x-X_PARAM[1])^X_PARAM[2]- Y_PARAM[0]*(y-Y_PARAM[1])^Y_PARAM[2]+ X_PARAM[3]+ X_PARAM[3] # X_PARAMS 1 0 2; Y_PARAMS 1 0 2
#         return z
#     elif choose==1:
#         print('sin(sqrt({0}*(x-{1})^{2})+{3}+ {4}*(y-{5})^{6}+{7} '.format(X_PARAM[0], X_PARAM[1], X_PARAM[2], X_PARAM[3], Y_PARAM[0], Y_PARAM[1], Y_PARAM[2], Y_PARAM[3]  ))
#         return np.sin(np.sqrt( X_PARAM[0]*(x-X_PARAM[1])**X_PARAM[2]+ Y_PARAM[0]*(y-Y_PARAM[1])**Y_PARAM[2] ))+X_PARAM[3]+ X_PARAM[3]


def calc_func(axis, fix, alpha):
    if axis == 'x':
        f = func(fix, alpha)
    elif axis == 'y':
        f = func(alpha, fix)
    else:
        raise NameError('Not a standard form(x or y)')
    return f


def visualize(x, y):
    x_surf = np.linspace(scope_x[0], scope_x[1], num)
    y_surf = np.linspace(scope_y[0], scope_y[1], num)
    xy = np.meshgrid(x_surf, y_surf)
    z_surf = func(xy[0], xy[1])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(xy[0], xy[1], z_surf,
                    cmap=cm.jet,
                    linewidth=0.2)
    z = func(x, y)
    ax.scatter(x, y, z)

    plt.show()


def show(track_x, track_y, track_z):
    for i in range(len(track_x)):
        track_z = func(np.array(track_x), np.array(track_y)).tolist()

        x_surf = np.linspace(scope_x[0] - 0.2, scope_x[1] + 0.2, num_divs)
        y_surf = np.linspace(scope_y[0] - 0.2, scope_y[1] + 0.2, num_divs)
        xy = np.meshgrid(x_surf, y_surf)
        z_surf = func(xy[0], xy[1])

        fig = plt.figure(figsize=(5, 5))
        # ax = fig.gca(projection='3d')
        ax = Axes3D(fig)
        ax.plot_trisurf(xy[0].reshape(-1), xy[1].reshape(-1), z_surf.reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=0.6)
        ax.set_title('Step %d' % (i + 1), fontsize=18)
        ax.view_init(elev=35., azim=185)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.scatter(track_x[i], track_y[i], track_z[i], c='black', s=43)
        # plt.savefig(r'F:\data\Experimental\temp_scatter' + str(i))

        print("No.%d finish" % (i))


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

    ax.set_title(r'Orthogonal Rotation Analysis $f(x,y)$', fontdict={'fontsize': 20, 'fontweight': 'medium'})
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)

    # ax.plot(final_x, final_y, marker='*', c='r', markersize=15)
    ax.scatter(final_x, final_y, marker='*', c='r', s=50, label='Final Result')

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

    ax.set_title(r'Orthogonal Rotation Analysis $f(x,y)$ 3D', fontdict={'fontsize': 20, 'fontweight': 'medium'})
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)

    # ax.plot([final_x], [final_y], [final_z], marker='*', c='r', markersize=20, label='Final result')
    ax.scatter([final_x], [final_y], [final_z], marker='*', c='r', s=80, label='Final result')

    # ax.plot(track_x[0], track_y[0], c='b', marker='o')

    return ax


def init3D():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])

    point1.set_data([], [])
    point1.set_3d_properties([])

    point2.set_data([], [])
    point2.set_3d_properties([])

    point3.set_data([], [])
    point3.set_3d_properties([])

    display_value.set_text('')

    # return line, point, display_value
    return line, point, point1, point2, point3, display_value


def animate3D(i):
    line.set_data(track_x[i], track_y[i])
    line.set_3d_properties(track_z[i])

    point.set_data(track_x[i], track_y[i])
    point.set_3d_properties(track_z[i])

    point1.set_data(track_x[i], track_y[i])
    point1.set_3d_properties([-0.5] * len(track_x[i]))

    # point2.set_data(scope_x[0]* len(track_x[i]), track_y[i]) # a bug here
    # point2.set_3d_properties(track_z[i])
    point2.set_data([scope_x[0] - 0.3] * len(track_x[i]), track_y[i])
    point2.set_3d_properties(track_z[i])

    point3.set_data(track_x[i], [scope_y[1] + 0.3] * len(track_x[i]))
    point3.set_3d_properties(track_z[i])

    display_value.set_text('Iteration: ' + str(i + 1) +
                           '\n'
                           r'$x(\alpha_1)= $' '{0:.4f}'.format(track_x[i][1]) + r' $x(\alpha_2)= $' '{0:.4f}'.format(
        track_x[i][2]) +
                           '\n'
                           r'$y(\alpha_1)= $' '{0:.4f}'.format(track_y[i][1]) + r' $y(\alpha_2)= $' '{0:.4f}'.format(
        track_y[i][2]) +
                           '\n'
                           r'$z(\alpha_1)= $' '{0:.4f}'.format(track_z[i][1]) + r' $z(\alpha_2)= $' '{0:.4f}'.format(
        track_z[i][2]))

    # return line, point, display_value
    return line, point, point1, point2, point3, display_value


def init2D():
    line.set_data([], [])
    point.set_data([], [])
    value_display.set_text('')
    return line, point, value_display


def animate2D(i):
    line.set_data(track_x[i], track_y[i])

    point.set_data(track_x[i], track_y[i])
    # point.set_data(track_x[i], track_y[i], track_z[i])

    value_display.set_text('Iteration: ' + str(i + 1) +
                           '\n'
                           r'$x(\alpha_1)= $' '{0:.4f}'.format(
                               track_x[i][1]) + r' $x(\alpha_2)= $' '{0:.4f}'.format(
        track_x[i][2]) +
                           '\n'
                           r'$y(\alpha_1)= $' '{0:.4f}'.format(
                               track_y[i][1]) + r' $y(\alpha_2)= $' '{0:.4f}'.format(
        track_y[i][2]) +
                           '\n'
                           r'$z(\alpha_1)= $' '{0:.4f}'.format(
                               track_z[i][1]) + r' $z(\alpha_2)= $' '{0:.4f}'.format(
        track_z[i][2])
                           )

    return line, point, value_display


parser = argparse.ArgumentParser(description=u'Orthogonal Rotation/Spin-up Method Final')
parser.add_argument('--x_min', help='left boundary of X ', type=int, default=0)
parser.add_argument('--x_max', help='right boundary of X', type=int, default=8)
parser.add_argument('--y_min', help='left boundary of Y', type=int, default=0)
parser.add_argument('--y_max', help='right boundary of Y', type=int, default=8)
parser.add_argument('--delta', help='delta', type=float, default=0.618)
parser.add_argument('--init_x', help='initial x', type=float, default=1.)
parser.add_argument('--init_y', help='initial y', type=float, default=0.)
parser.add_argument('--tol', help='tolerance', type=float, default=0.0001)

# parser.add_argument('--num', help='delta', type=float, default=0.)
parser.add_argument('--num_divs', help='num of divides plot', type=float, default=50)

# X_PARAM=[1., 0., 1., 0., 0, 0, 0, 0, float('nan'),float('nan')]
# Y_PARAM=[1., 0., 1., 0., 0, 0, 0, 0, float('nan'),float('nan')]
# CHOOSE=-1
parser.add_argument('--CHOOSE', help='to choose function', type=int, default=-1)
parser.add_argument('--x_p1', help='parmeter1 of x, eg:k1*x', type=float, default=0.)
parser.add_argument('--x_p2', help='parmeter2 of x, eg:k1*(x-k2)', type=float, default=0.)
parser.add_argument('--x_p3', help='parmeter3 of x, eg:k1*(x-k2)^k3', type=float, default=1.0)
parser.add_argument('--x_p4', help='parmeter4 of x, eg:k1*(x-k2)^k3+k4', type=float, default=0)
parser.add_argument('--x_p5', help='parmeter5 of x, eg:k5*x**-1', type=float, default=0.)
# parser.add_argument('--x_p6', help='parmeter6 of x, eg:k6*x**6', type=float, default=0.)
# parser.add_argument('--x_p7', help='parmeter7 of x, eg:k6*x**7', type=float, default=0.)
# parser.add_argument('--x_p8', help='parmeter8 of x, eg:k8*x**8', type=float, default=0.)
parser.add_argument('--y_p1', help='parmeter1 of y, eg:k1*x', type=float, default=1.)
parser.add_argument('--y_p2', help='parmeter2 of y, eg:k1*(y-k2)', type=float, default=0.)
parser.add_argument('--y_p3', help='parmeter3 of y, eg:k1*(y-k2)^k3', type=float, default=1.0)
parser.add_argument('--y_p4', help='parmeter4 of y, eg:k1*(y-k2)^k3+k4', type=float, default=0)
parser.add_argument('--y_p5', help='parmeter5 of y, eg:k5*x**-1', type=float, default=0.)

if __name__ == '__main__':
    args = parser.parse_args()
    x_min = args.x_min
    x_max = args.x_max
    y_min = args.y_min
    y_max = args.y_max
    delta = args.delta

    init_x = args.init_x
    init_y = args.init_y
    tol = args.tol
    num_divs = args.num_divs

    CHOOSE = args.CHOOSE
    print('CHOOSE:',CHOOSE)

    X_PARAM[0] = x_p1 = args.x_p1
    X_PARAM[1] = x_p2 = args.x_p2
    X_PARAM[2] = x_p3 = args.x_p3
    X_PARAM[3] = x_p4 = args.x_p4
    X_PARAM[4] = x_p5 = args.x_p5

    Y_PARAM[0] = y_p1 = args.y_p1
    Y_PARAM[1] = y_p2 = args.y_p2
    Y_PARAM[2] = y_p3 = args.y_p3
    Y_PARAM[3] = y_p4 = args.y_p4
    Y_PARAM[4] = y_p5 = args.y_p5

    if CHOOSE == -1:
        print(
            'f(x,y) = {0}*(x-{1})^{2}+{3}+ {4}*(y-{5})^{6}+{7} '.format(X_PARAM[0], X_PARAM[1], X_PARAM[2], X_PARAM[3],
                                                                        Y_PARAM[0], Y_PARAM[1], Y_PARAM[2],
                                                                        Y_PARAM[3]))
    elif CHOOSE == 0:
        print('f(x,y) = sin(sqrt({0}*(x-{1})^{2})+{3}+ {4}*(y-{5})^{6}+{7} '.format(X_PARAM[0], X_PARAM[1], X_PARAM[2],
                                                                                    X_PARAM[3], Y_PARAM[0], Y_PARAM[1],
                                                                                    Y_PARAM[2], Y_PARAM[3]))
    else:
        print('-x**2 - y**2')

    final_x, final_y, final_z = union_factor(init_x, init_y)
    track_z = func(np.array(track_x), np.array(track_y))

    history = {}
    history['x'] = track_x
    history['y'] = track_y
    history['z'] = track_z
    pkl('history.pkl', history)

    # 2D
    fig = plt.figure(figsize=(8, 8))
    # ax1 = Axes3D(fig)
    ax1 = plt.gca()
    ax1 = snapshot2D(ax1)

    line, = ax1.plot([], [], 'r', label='Domain', lw=1.5)
    point, = ax1.plot([], [], 'bo', c='b', label='Experiments')
    value_display = ax1.text(0.02, 0.08, '', transform=ax1.transAxes)

    ax1.legend(loc=1)
    ax1.grid(False)

    anim2D = animation.FuncAnimation(fig, animate2D, init_func=init2D,
                                     frames=len(track_x),
                                     interval=500,
                                     repeat_delay=80, blit=True)

    Writer2D = animation.writers['ffmpeg']
    FPS_2D = 5
    writer2D = Writer2D(fps=FPS_2D, metadata=dict(artist='L'), bitrate=1800)
    # anim.save('Orthogonal-Rotation-004.gif',writer='imagemagick', fps=60)
    print('Creating 2D Animation VIDEO ...')
    anim2D.save('Orthogonal-Rotation-2D-fps{0}.mp4'.format(FPS_2D), writer=writer2D)
    print('Done.\nCreating 2D Animation GIF ...')
    anim2D.save('Orthogonal-Rotation-2D-fps{0}.gif'.format(FPS_2D), writer='imagemagick', fps=FPS_2D)
    print('Done\n2D Animation Created.')
    plt.savefig('Orthogonal-Rotation-2D.pdf')
    plt.savefig('Orthogonal-Rotation-2D.png')

    # 3D
    fig1 = plt.figure(figsize=(8, 8))
    ax2 = Axes3D(fig1)
    ax2 = snapshot3D(ax2)

    line, = ax2.plot([], [], [], 'r-', label='Domain', lw=1.5)
    point, = ax2.plot([], [], [], 'bo', markersize=10, label='Experiments')  # on the surface
    point1, = ax2.plot([], [], [], 'ro', markersize=10, alpha=0.2)  # projection to z
    point2, = ax2.plot([], [], [], 'ro', markersize=10, alpha=0.2)  # projection to x
    point3, = ax2.plot([], [], [], 'ro', markersize=10, alpha=0.2)  # projection to y

    display_value = ax2.text(scope_x[1], scope_y[1], final_z + 0.5, '', transform=ax2.transAxes)
    ax2.grid(False)
    ax2.legend(loc=1)

    anim = animation.FuncAnimation(fig1, animate3D, init_func=init3D,
                                   frames=len(track_x),
                                   interval=1000,
                                   repeat_delay=80, blit=True)

    Writer3D = animation.writers['ffmpeg']
    FPS_3D = 5
    writer3D = Writer3D(fps=FPS_3D, metadata=dict(artist='L'), bitrate=1800)
    # anim.save('Orthogonal-Rotation-004.gif',writer='imagemagick', fps=6)
    print('Creating 3D Animation VIDEO ...')
    anim.save('Orthogonal-Rotation-3D-fps{0}.mp4'.format(FPS_3D), writer=writer3D)
    print('Done.\nCreating 3D Animation GIF ...')
    anim.save('Orthogonal-Rotation-3D-fps{0}.gif'.format(FPS_3D), writer='imagemagick', fps=FPS_3D)
    print('Done\n3D Animation Created.')
    plt.savefig('Orthogonal-Rotation-3D.pdf')
    plt.savefig('Orthogonal-Rotation-3D.png')
    # plt.show()
