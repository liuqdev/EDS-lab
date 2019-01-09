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

sns.set()


def union_factor(init_x, init_y):
    '''
    '''
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
    '''
    '''

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
    '''
    '''
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
    '''
    '''
    gap = bord_right - bord_left

    alpha_0 = bord_right - delta * gap
    alpha_1 = bord_left + delta * gap

    return alpha_0, alpha_1


def func(x, y):
    '''
    '''
    z = (x - 4) * np.exp(-(((x - 5) / 2.) ** 2 + ((y - 4.5) / 1.5) ** 2))
    return z


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


def snapshot(ax):

    a = np.linspace(scope_x[0] - 0.2, scope_x[1] + 0.2, num_divs)
    b = np.linspace(scope_y[0] - 0.2, scope_y[1] + 0.2, num_divs)
    x, y = np.meshgrid(a, b)
    # z=func(x,y)
    # ax.contour(x, y,z, levels=np.logspace(-3,3,25), cmap='jet')
    ax.contourf(x, y, func(x, y), 100, cmap=cm.hot, alpha=0.7)
    CS = ax.contour(x, y, func(x, y), colors='k')
    ax.clabel(CS, inline=True, fontsize=13)
    ax.set_xlabel(r'x', fontdict={'fontsize': 18, 'fontweight': 'medium'})
    ax.set_ylabel(r'y', fontdict={'fontsize': 18, 'fontweight': 'medium'})

    ax.set_title(r'Orthogonal Rotation Analysis $f(x,y)$', fontdict={'fontsize': 20, 'fontweight': 'medium'})
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)

    ax.plot(final_x, final_y, marker='*', c='r', markersize=15)

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

    #ax.plot([final_x], [final_y], [final_z], marker='*', c='r', markersize=20, label='Final result')
    ax.scatter([final_x], [final_y], [final_z], marker='*', c='r', s=80, label='Final result')

    # ax.plot(track_x[0], track_y[0], c='b', marker='o')

    return ax


def init():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])

    point1.set_data([], [])
    point1.set_3d_properties([])

    display_value.set_text('')

    #return line, point, display_value
    return line, point,point1, display_value


def animate(i):

    line.set_data(track_x[i], track_y[i])
    line.set_3d_properties(track_z[i])

    point.set_data(track_x[i], track_y[i])
    point.set_3d_properties(track_z[i])

    point1.set_data(track_x[i], track_y[i])
    point1.set_3d_properties([-0.5]*len(track_x[i]))


    display_value.set_text('Iteration: '+str(i+1)+
                           '\n'
                           r'$x(\alpha_1)= $' '{0:.4f}'.format(track_x[i][1])+r' $x(\alpha_2)= $' '{0:.4f}'.format(track_x[i][2])+
                           '\n'
                           r'$y(\alpha_1)= $' '{0:.4f}'.format(track_y[i][1])+r' $y(\alpha_2)= $' '{0:.4f}'.format(track_y[i][2])+
                           '\n'
                           r'$z(\alpha_1)= $' '{0:.4f}'.format(track_z[i][1])+r' $z(\alpha_2)= $' '{0:.4f}'.format(track_z[i][2]))

    #return line, point, display_value
    return line, point, point1, display_value




##############################
delta = 0.618
x_min = 0
x_max = 8
y_min = 0
y_max = 8
num = 30
num_divs = 50
tol = 0.0001

init_x = 1
init_y = 0

final_x = None
final_y = None
final_z = None

scope_x = [float(x_min), float(x_max)]
scope_y = [float(y_min), float(y_max)]
track_x = []
track_y = []
#############################

parser = argparse.ArgumentParser(description=u'Orthogonal Rotation Analysis')
parser.add_argument('--x_min', help='x_min', type=int, default=0)
parser.add_argument('--x_max', help='x_max', type=int, default=8)
parser.add_argument('--y_min', help='y_min', type=int, default=0)
parser.add_argument('--y_max', help='y_max', type=int, default=8)


if __name__ == '__main__':
    args = parser.parse_args()
    x_min = args.x_min
    x_max = args.x_max
    y_min = args.y_min
    y_max = args.y_max


    final_x, final_y, final_z = union_factor(init_x, init_y)
    track_z = func(np.array(track_x), np.array(track_y))

    fig1 = plt.figure(figsize=(12, 12))
    ax2 = Axes3D(fig1)
    ax2 = snapshot3D(ax2)

    line, = ax2.plot([], [], [], 'r-', label='Domain', lw=1.5)
    point, = ax2.plot([], [], [], 'ro', markersize=10, label='Experiments')
    point1, = ax2.plot([], [], [], 'ro', markersize=10, alpha=0.2)
    display_value = ax2.text(scope_x[1], scope_y[1], final_z + 0.5, '', transform=ax2.transAxes)

    ax2.legend(loc=1)

    anim = animation.FuncAnimation(fig1, animate, init_func=init,
                                   frames=len(track_x),
                                   interval=500,
                                   repeat_delay=80, blit=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='L'), bitrate=1800)
    # anim.save('Orthogonal-Rotation-004.gif',writer='imagemagick', fps=60)
    print('Creating Animation VIDEO ...')
    anim.save('Orthogonal-Rotation-3D.mp4', writer=writer)
    print('Done.\nCreating Animation GIF ...')
    anim.save('Orthogonal-Rotation-3D.gif', writer='imagemagick', fps=15)
    print('Done\nAnimation Created.')
    # plt.show()


