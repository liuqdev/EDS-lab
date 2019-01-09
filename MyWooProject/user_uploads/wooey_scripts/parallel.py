import argparse
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import matplotlib.patches as patches

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

# x_low =4
# x_high = 8
# y_low = -4
# y_high = 8
#
# x_range = x_high - x_low
# y_range = y_high - y_low
# error = 1
def fun(x,y):
    #z = -x ** 2 - y ** 2 + 2 * x + y + 5
    z = (x - 4) * np.exp(-(((x - 5) / 2.) ** 2 + ((y - 4.5) / 1.5) ** 2))
    return z

def yhighf(xmin, xmax, ymin, ymax):
    x_low = xmin
    x_high = xmax
    y_low = ymin
    y_high = ymax

    range = x_high - x_low
    x_low_try = x_low + (1 - 0.618) * range
    x_high_try = x_low + 0.618 * range

    step = 0
    while step < 20:
        step = step + 1
        z_low_try = fun(x_low_try, y_high)
        z_high_try = fun(x_high_try, y_high)

        if z_low_try > z_high_try:
            x_high = x_high_try
            range = x_high - x_low
            x_low_try = x_low + (1 - 0.618) * range
            x_high_try = x_low + 0.618 * range

        else:
            x_low = x_low_try
            range = x_high - x_low
            x_low_try = x_low + (1 - 0.618) * range
            x_high_try = x_low + 0.618 * range

        if step == 20:
            z = (z_low_try + z_high_try)/2
            x = (x_high + x_low)/2
    return  x,y_high,z, (x_low_try + x_high_try) / 2, y_high


def ylowf(xmin, xmax, ymin, ymax):
    x_low = xmin
    x_high = xmax
    y_low = ymin
    y_high = ymax

    range = x_high - x_low
    x_low_try = x_low + (1 - 0.618) * range
    x_high_try = x_low + 0.618 * range

    step = 0
    while step < 20:
        step = step + 1
        z_low_try = fun(x_low_try, ymin)
        z_high_try = fun(x_high_try, ymin)

        if z_low_try > z_high_try:
            x_high = x_high_try
            range = x_high - x_low
            x_low_try = x_low + (1 - 0.618) * range
            x_high_try = x_low + 0.618 * range

        else:
            x_low = x_low_try
            range = x_high - x_low
            x_low_try = x_low + (1 - 0.618) * range
            x_high_try = x_low + 0.618 * range

        if step == 20:
            z = (z_low_try + z_high_try)/2
            x = (x_high + x_low)/2
    return  x,y_low,z, (x_low_try + x_high_try) / 2,ymin


parser = argparse.ArgumentParser(description=u'Parallel Method Visulization')
args = parser.parse_args()


print("xf:",yhighf(-4,4,-4,4))
print("yf:",ylowf(-4,4,-4,4))

times = 0

range = y_high - y_low
y_low_try = y_low + (1 - 0.618) * range
y_high_try = y_low + 0.618 * range

history = {}
history['domain'] = []
history['x_best'] = []
history['y_best'] = []
while times < 10 :
    times = times +1
    print('\nIteration', times)

    x_yhigh,y_yhigh,z_yhigh, xofx, yofx = yhighf(x_low,x_high,y_low_try,y_high_try)
    x_ylow, y_ylow, z_ylow, xofy, yofy  = ylowf(x_low, x_high, y_low_try, y_high_try)
    if z_yhigh < z_ylow:
        y_high = y_high_try
        range = y_high - y_low
        y_low_try = y_low + (1 - 0.618) * range
        y_high_try = y_low + 0.618 * range

    else:
        y_low = y_low_try
        range = y_high - y_low
        y_low_try = y_low + (1 - 0.618) * range
        y_high_try = y_low + 0.618 * range

    print('z_yhigh', z_yhigh)
    print('z_ylow',z_ylow)
    print('x_yhigh',x_yhigh)
    print('x_ylow',x_ylow)
    print('y_yhigh',y_yhigh)
    print('y_ylow',y_ylow)
    history['domain'].append([x_low, x_high, y_low, y_high])
    history['x_best'].append([xofx, yofx, z_yhigh])
    history['y_best'].append([xofy, yofy, z_ylow])



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

    # x_width = domain_t[1] - domain_t[0]
    # y_width = domain_t[3] - domain_t[2]
    # x_rect, y_rect = domain_t[0], domain_t[2]  # left corner

    # x_rect, y_rect = (x_high+x_low)/2,  (y_high+y_low)/2  # left corner
    # print(x_rect, y_rect)
    #ax.add_patch(Rectangle((x_rect, y_rect), x_width, y_width, alpha=0.6, color='w'))
    # ax.scatter(x_best_t[0], x_best_t[1], s=50, c='r', alpha=1, zorder=9) # y fixed
    # ax.plot([4, 8], [2, 2], 'k--', lw=2.5)
    # print([domain_t[0], domain_t[1]], [x_best_t[1], x_best_t[1]])
    # ax.plot([domain_t[0], domain_t[1],], [x_best_t[1], x_best_t[1],], 'r--', lw=1.5)
    # ax.scatter(y_best_t[0], y_best_t[1], s=50, c='b', alpha=1, zorder=9) # x fixed
    # ax.plot([y_best_t[0], y_best_t[0],], [domain_t[2], domain_t[3],], 'b--', lw=1.5)

    return ax


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


# pkl('history-tws.pkl', history)
# history=unpkl('history-tws.pkl')

# print(len(history['domain']), len(history['x_best']), len(history['x_best']))
print(len(history['domain']))
ti=0
domain_t=history['domain'][ti]
# x_best_t=history['x_best'][ti]
# y_best_t=history['y_best'][ti]
print(domain_t)



fig = plt.figure()

ax=fig.add_subplot(111)
# ax = fig.add_subplot(111, projection='3d')
# ax = Axes3D(fig)
ax=snapshot(ax)


# line_h, = ax.plot([], [], 'r--', label='y fixed', lw=1.5)
# line_v, = ax.plot([], [], 'b--', label='x fixed ', lw=1.5) # x fixed
line_h, = ax.plot([], [], 'r--', label='high boundary', lw=1.5)
line_v, = ax.plot([], [], 'b--', label='low boundary ', lw=1.5) # x fixed
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
    # return  value_display, patch


def animate(i):

    domain_t = history['domain'][i]
    x_best_t = history['x_best'][i]
    y_best_t = history['y_best'][i]
    # print(domain_t)

    line_h.set_data([domain_t[0], domain_t[1],], [x_best_t[1], x_best_t[1],])
    # line_v.set_data([y_best_t[0], y_best_t[0],], [domain_t[2], domain_t[3],])
    line_v.set_data([domain_t[0], domain_t[1],], [y_best_t[1], y_best_t[1], ])
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
    # return value_display, patch

ax.legend(loc=1)

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(history['domain']),
                               interval=500,
                               repeat_delay=80, blit=True)

plt.grid(False)


Writer=animation.writers['ffmpeg']
writer=Writer(fps=2, metadata=dict(artist='L'), bitrate=1800)
anim.save('parallel-001.mp4', writer=writer)

anim.save('parallel-001.mp4',writer='imagemagick')
anim.save('parallel-001.gif',writer='imagemagick',fps=2)
plt.savefig('parallel-001.pdf')
plt.savefig('parallel-001.png')
# plt.show()




