import argparse
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.dpi'] = 120
# plt.rcParams['axes.axisbelow'] = True
from matplotlib import cm
import seaborn as sns
# sns.set_style('white')

#########################################
x_lim = [4, 8]  # x_limits of plot
y_lim = [-4, 8]  # y_limits of plot

# searching boundaries
x_low = 4
x_high = 8
x_step = 0.1
y_low = -4
y_high = 8
y_step = 0.1

x_range = x_high - x_low
y_range = y_high - y_low
error = 1


#########################################

def fun(x, y):
    # z = -x ** 2 - y ** 2 + 2 * x + y + 5
    z = (x - 4) * np.exp(-(((x - 5) / 2.) ** 2 + ((y - 4.5) / 1.5) ** 2))
    return z


#
def xf(xmin, xmax, ymin, ymax):
    x_low = xmin
    x_high = xmax
    y_low = ymin
    y_high = ymax

    x_middle = (x_low + x_high) / 2
    y_middle = (y_low + y_high) / 2
    print('--------Keep y = {0} fixed--------'.format(y_middle))

    x_range = x_high - x_low
    x_low_try = x_low + (1 - 0.618) * x_range
    x_high_try = x_low + 0.618 * x_range

    step = 0
    while step < 20:
        step = step + 1
        z_low_try = fun(x_low_try, y_middle)
        z_high_try = fun(x_high_try, y_middle)

        if z_low_try > z_high_try:
            x_high = x_high_try
            x_range = x_high - x_low
            x_low_try = x_low + (1 - 0.618) * x_range
            x_high_try = x_low + 0.618 * x_range

        else:
            x_low = x_low_try
            x_range = x_high - x_low
            x_low_try = x_low + (1 - 0.618) * x_range
            x_high_try = x_low + 0.618 * x_range

        if step >= 20:
            result = (x_low_try + x_high_try) / 2
            xz = (z_low_try + z_high_try) / 2
            if result < x_middle:
                x_low = xmin
                x_high = x_middle
            else:
                x_low = x_middle
                x_high = xmax

    print('Find best z = {0:.4f} at ({1:.4f},{2:.4f})\tNew [x_low, x_high] = [{3:.3f},{4:.4f}]'.format(xz, result,
                                                                                                       y_middle, x_low,
                                                                                                       x_high))
    return x_low, x_high, xz, result, y_middle


def yf(xmin, xmax, ymin, ymax):
    x_low = xmin
    x_high = xmax
    y_low = ymin
    y_high = ymax

    x_middle = (x_low + x_high) / 2
    y_middle = (y_low + y_high) / 2
    print('--------Keep x = {0} fixed--------'.format(x_middle))

    y_range = y_high - y_low
    y_low_try = y_low + (1 - 0.618) * y_range
    y_high_try = y_low + 0.618 * y_range

    step = 0
    while step < 20:
        step = step + 1
        z_low_try = fun(x_middle, y_low_try)
        z_high_try = fun(x_middle, y_high_try)

        if z_low_try > z_high_try:
            y_high = y_high_try
            y_range = y_high - y_low
            y_low_try = y_low + (1 - 0.618) * y_range
            y_high_try = y_low + 0.618 * y_range

        else:
            y_low = y_low_try
            y_range = y_high - y_low
            y_low_try = y_low + (1 - 0.618) * y_range
            y_high_try = y_low + 0.618 * y_range

        if step >= 20:
            result = (y_low_try + y_high_try) / 2
            yz = (z_low_try + z_high_try) / 2
            if result < y_middle:
                y_low = ymin
                y_high = y_middle
            else:
                y_low = y_middle
                y_high = ymax
    # print('Find best z={0:.4f} at {1:.4f,2:.4f}')
    print('Find best z = {0:.4f} at ({1:.4f},{2:.4f})\tNew [y_low, y_high] = [{3:.4f},{4:.4f}]'.format(yz, x_middle,
                                                                                                       result, y_low,
                                                                                                       y_high))
    return y_low, y_high, yz, x_middle, result

#
# a = np.arange(x_lim[0], x_lim[1], x_step)
# b = np.arange(y_lim[0], y_lim[1], y_step)
# X, Y = np.meshgrid(a, b)
# Z = fun(X, Y)
# fig = plt.figure()
# # ax = Axes3D(fig)
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim(x_lim)
# ax.set_ylim(y_lim)
# # ax.scatter(X, Y, Z)
# ax.plot_surface(X, Y, Z, cmap='jet')
# plt.show()

# x_lo_init, x_hi_init = -4, 4
# y_lo_init, y_hi_init = -4, 4

print('\n---------Searching Filed----------')
print('[x_low, x_high] = [{0:.4f},{1:.4f}]'.format(x_low, x_high))
print('[y_low, y_high] = [{0:.4f},{1:.4f}]'.format(y_low, y_high))

# print('\n---------Iteration: {0:0>3d}-----------'.format(0))
# px_init=xf(x_lo_init, x_hi_init, y_lo_init, y_hi_init)
# py_init=yf(x_lo_init, x_hi_init, y_lo_init, y_hi_init)

# print('x_low, x_high {0:.4f},{1:.4f}'.format(px_init[0], px_init[1]))
# print('zx {0:.4f}'.format(px_init[2]))
# print('y_low, y_high {0:.4f},{1:.4f}'.format(py_init[0], py_init[1]))
# print('zy {0:.4f}'.format(py_init[2]))

# print("New x domain({0},{1}) value {2:.4f}".format(px_init[0], px_init[1], px_init[2]) )
# print("New y domain({0},{1}) value {2:.4f}".format(py_init[0], py_init[1], py_init[2])  )





parser = argparse.ArgumentParser(description=u'Two Ways Divide Method Final')
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

parser.add_argument('--x_step', help='step of X ', type=float, default=0.1)
parser.add_argument('--y_step', help='step of Y', type=float, default=0.1)
parser.add_argument('--error', help='error', type=int, default=1)
parser.add_argument('--max_iter', help='max_iter', type=int, default=20)

parser.add_argument('--tol_x', help='tolerance of X', type=float, default=0.001)
parser.add_argument('--tol_y', help='tolerance of Y', type=float, default=0.001)
# X_PARAM=[1., 0., 1., 0., 0, 0, 0, 0, float('nan'),float('nan')]
# Y_PARAM=[1., 0., 1., 0., 0, 0, 0, 0, float('nan'),float('nan')]
# CHOOSE=-1
# parser.add_argument('--CHOOSE', help='to choose function', type=int, default=-1)
# parser.add_argument('--x_p1', help='parmeter1 of x, eg:k1*x', type=float, default=0.)
# parser.add_argument('--x_p2', help='parmeter2 of x, eg:k1*(x-k2)', type=float, default=0.)
# parser.add_argument('--x_p3', help='parmeter3 of x, eg:k1*(x-k2)^k3', type=float, default=1.0)
# parser.add_argument('--x_p4', help='parmeter4 of x, eg:k1*(x-k2)^k3+k4', type=float, default=0)
# parser.add_argument('--x_p5', help='parmeter5 of x, eg:k5*x**-1', type=float, default=0.)
# # parser.add_argument('--x_p6', help='parmeter6 of x, eg:k6*x**6', type=float, default=0.)
# # parser.add_argument('--x_p7', help='parmeter7 of x, eg:k6*x**7', type=float, default=0.)
# # parser.add_argument('--x_p8', help='parmeter8 of x, eg:k8*x**8', type=float, default=0.)
# parser.add_argument('--y_p1', help='parmeter1 of y, eg:k1*x', type=float, default=1.)
# parser.add_argument('--y_p2', help='parmeter2 of y, eg:k1*(y-k2)', type=float, default=0.)
# parser.add_argument('--y_p3', help='parmeter3 of y, eg:k1*(y-k2)^k3', type=float, default=1.0)
# parser.add_argument('--y_p4', help='parmeter4 of y, eg:k1*(y-k2)^k3+k4', type=float, default=0)
# parser.add_argument('--y_p5', help='parmeter5 of y, eg:k5*x**-1', type=float, default=0.)


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


x_lim = [x_min, x_max]  # x_limits of plot
y_lim = [y_min, y_max]  # y_limits of plot

# searching boundaries
# x_low = x_min
# x_high = x_max
x_step = args.x_step
y_low = args.y_step
error = args.error
# y_high = y_min
# y_step = y_max
#
# x_range = x_high - x_low
# y_range = y_high - y_low
max_iter = args.max_iter
tol_x = args.tol_x
tol_y = args.tol_y
# max_iter = 20
# tol_x = 0.001
# tol_y = 0.001
history = {}
history['domain'] = []
history['x_best'] = []
history['y_best'] = []
for i in range(0, max_iter):
    if abs(x_high - x_low) < tol_x and abs(y_high - y_low) < tol_y:
        break

    print('\n---------Iteration: {0:0>3d}-----------'.format(i + 1))
    history['domain'].append([x_low, x_high, y_low, y_high])
    x_new_low, x_new_high, zx, xofx, yofx = xf(x_low, x_high, y_low, y_high)
    y_new_low, y_new_high, zy, xofy, yofy = yf(x_low, x_high, y_low, y_high)
    # history['domain'].append([x_new_low, x_new_high, y_new_low, y_new_high])
    history['x_best'].append([xofx, yofx, zx])
    history['y_best'].append([xofy, yofy, zy])

    x_low = x_new_low
    x_high = x_new_high
    y_low = y_new_low
    y_high = y_new_high


    # print('zx {0:.4f}'.format(zx))
    # print('zy {0:.4f}'.format(zy))
    # print('x_low, x_high {0:.4f},{1:.4f}'.format(x_low, x_high))
    # print('y_low, y_high {0:.4f},{1:.4f}'.format(y_low, y_high))

# print('Search History')
# print(history)

# someX, someY = 0.5, 0.5
# fig=plt.figure(figsize=(6,6))
# ax=fig.add_subplot(111)
# ax.add_patch(Rectangle((someX - .1, someY - .1), 0.2, 0.2,alpha=1))
# plt.show()

def pkl(path, d):
    import pickle
    with open(path, 'wb') as fo:
        pickle.dump(d, fo)
        print('SUSS')

def unpkl(path):
    import pickle
    with open(path, 'rb') as fo:
        d=pickle.load(fo)
        print('SUSS')
    return d

pkl('history-tws.pkl', history)
history=unpkl('history-tws.pkl')

#print(len(history['domain']), len(history['x_best']), len(history['x_best']))
# ti=0
# domain_t=history['domain'][ti]
# x_best_t=history['x_best'][ti]
# y_best_t=history['y_best'][ti]
# print(domain_t)

fig=plt.figure(figsize=(6,6))
ax=fig.add_subplot(111)

def snapshot(ax):
    a = np.arange(x_lim[0]-0.3, x_lim[1]+0.3, x_step)
    b = np.arange(y_lim[0]-0.3, y_lim[1]+0.3, y_step)
    X, Y = np.meshgrid(a, b)
    Z = fun(X, Y)
    # ax.contourf(X, Y, Z, 10,  alpha=0.1) # fill
    # ax.contourf(X, Y, Z, 40, alpha=0.2, cmap='viridis')
    ax.contourf(X, Y, Z, 40, alpha=0.7, cmap='viridis')
    CS = ax.contour(X, Y, Z, 15, colors='w') # add contour lines
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_xlabel(r'X', fontdict={'fontsize': 10, 'fontweight': 'medium'})
    ax.set_ylabel(r'Y', fontdict={'fontsize': 10, 'fontweight': 'medium'})

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # x_width = domain_t[1]-domain_t[0]
    # y_width = domain_t[3]-domain_t[2]
    # x_rect, y_rect = domain_t[0],  domain_t[2]  # left corner
    # #x_rect, y_rect = (x_high+x_low)/2,  (y_high+y_low)/2  # left corner
    # # print(x_rect, y_rect)
    #
    # ax.add_patch(Rectangle( (x_rect, y_rect), x_width, y_width ,alpha=0.6, color='w' ))
    # ax.scatter(x_best_t[0], x_best_t[1], s=50, c='r', alpha=1, zorder=9) # y fixed
    # # ax.plot([4, 8], [2, 2], 'k--', lw=2.5)
    # # print([domain_t[0], domain_t[1]], [x_best_t[1], x_best_t[1]])
    # ax.plot([domain_t[0], domain_t[1],], [x_best_t[1], x_best_t[1],], 'r--', lw=1.5)
    # ax.scatter(y_best_t[0], y_best_t[1], s=50, c='b', alpha=1, zorder=9) # x fixed
    # ax.plot([y_best_t[0], y_best_t[0],], [domain_t[2], domain_t[3],], 'b--', lw=1.5)
    return ax

ax=snapshot(ax)

line_h, = ax.plot([], [], 'r--', label='y fixed', lw=1.5)
line_v, = ax.plot([], [], 'b--', label='x fixed ', lw=1.5) # x fixed
point_a, = ax.plot([], [], 'bo', c='r', label='')
point_b, = ax.plot([], [], 'bo', c='b', label='')
value_display = ax.text(0.02, 0.08, '', transform=ax.transAxes)
patch=patches.Rectangle((0,0), 0, 0, alpha=0.6, color='w' )


def init():
    line_h.set_data([], [])
    line_v.set_data([], [])
    point_a.set_data([], [])
    point_b.set_data([], [])
    value_display.set_text('')
    ax.add_patch(patch)
    return line_h,line_v, point_a, point_b, value_display, patch

def animate(i):

    domain_t = history['domain'][i]
    x_best_t = history['x_best'][i]
    y_best_t = history['y_best'][i]
    # print(domain_t)

    line_h.set_data([domain_t[0], domain_t[1],], [x_best_t[1], x_best_t[1],])
    line_v.set_data([y_best_t[0], y_best_t[0],], [domain_t[2], domain_t[3],])
    point_a.set_data(x_best_t[0], x_best_t[1])
    point_b.set_data(y_best_t[0], y_best_t[1])

    x_width = domain_t[1] - domain_t[0]
    y_width = domain_t[3] - domain_t[2]
    x_rect, y_rect = domain_t[0], domain_t[2]  # left corner

    #ax.add_patch(Rectangle((x_rect, y_rect), x_width, y_width, alpha=0.6, color='w'))
    patch.set_width(x_width)
    patch.set_height(y_width)
    patch.set_xy([x_rect, y_rect])


    value_display.set_text('Iteration: '+str(i+1)+
                           '\n'
                           r'$Val(\alpha_1)= $' '{0:.4f}'.format(x_best_t[2])+' at ({0:.4f}, {1:.4f})'.format(x_best_t[0], x_best_t[1])+'\n'
                        r'$Val(\alpha_2)= $' '{0:.4f}'.format(x_best_t[2]) + ' at ({0:.4f}, {1:.4f})'.format(y_best_t[0], y_best_t[1])
                           )

    return line_h, line_v, point_a,point_b, value_display, patch

ax.legend(loc=1)

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(history['domain']),
                               interval=500,
                               repeat_delay=80, blit=True)

plt.grid(False)


Writer=animation.writers['ffmpeg']
writer=Writer(fps=2, metadata=dict(artist='L'), bitrate=1800)
anim.save('TwoWays-001.mp4', writer=writer)

# anim.save('TwoWays-001.mp4',writer='imagemagick')
anim.save('TwoWays-001.gif',writer='imagemagick',fps=2)
plt.savefig('TwoWays-001.pdf')
plt.savefig('TwoWays-001.png')
plt.show()
