import numpy as np

# # eg: -a(x-b)^c+d 1 0 1 1  -x+1
# X_PARAM=[1., 0., 1., 0., 0, 0, 0, float('nan'),float('nan')]
# Y_PARAM=[1., 0., 1., 0., 0, 0, 0, float('nan'),float('nan')]
# CHOOSE=-1  #
#
# def func(x, y, choose=CHOOSE):
#     if choose==-1:
#         return (x - 4) * np.exp(-(((x - 5) / 2.) ** 2 + ((y - 4.5) / 1.5) ** 2)) # just for testing, no parameters
#     elif choose==0:
#         print('x={0} y={1}'.format(x, y))
#         print(
#             'f(x,y) = {0}*(x-{1})^{2}+{3}+ {4}*(y-{5})^{6}+{7} '.format(X_PARAM[0], X_PARAM[1], X_PARAM[2], X_PARAM[3],
#                                                                          Y_PARAM[0], Y_PARAM[1], Y_PARAM[2],
#                                                                          Y_PARAM[3]))
#         z = X_PARAM[0]*(x-X_PARAM[1])**X_PARAM[2]+ Y_PARAM[0]*(y-Y_PARAM[1])**Y_PARAM[2]+ X_PARAM[3]+ X_PARAM[3] # X_PARAMS 1 0 2; Y_PARAMS 1 0 2
#         return z
#     elif choose==1:
#         print('x={0} y={1}'.format(x, y))
#         print('f(x,y) = sin(sqrt({0}*(x-{1})^{2})+{3}+ {4}*(y-{5})^{6}+{7} '.format(X_PARAM[0], X_PARAM[1], X_PARAM[2], X_PARAM[3], Y_PARAM[0], Y_PARAM[1], Y_PARAM[2], Y_PARAM[3]  ))
#         return np.sin(np.sqrt( X_PARAM[0]*(x-X_PARAM[1])**X_PARAM[2]+ Y_PARAM[0]*(y-Y_PARAM[1])**Y_PARAM[2] ))+X_PARAM[3]+ X_PARAM[3]
#     elif choose==-2: # not recomanded
#         z=(1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
#         #z1= X_PARAM[0]*(x-X_PARAM[1])**X_PARAM[2]+ X_PARAM[3] + X_PARAM[4]*x**4+X_PARAM[5]*x**5 +Y_PARAM[0]*(x-Y_PARAM[1])**Y_PARAM[2]+ Y_PARAM[3] + Y_PARAM[4]*x**4+Y_PARAM[5]*x**5
#         # z1 = ( X_PARAM[0] * (x - X_PARAM[1]) ** X_PARAM[2] + X_PARAM[3] + X_PARAM[4] * x ** 4 + X_PARAM[5] * x ** 5 + Y_PARAM[
#         #     0] * (x - Y_PARAM[1]) ** Y_PARAM[2] + Y_PARAM[3] + Y_PARAM[4] * x ** 4 + Y_PARAM[5] * x ** 5)*np.exp(-x**2-y**2)
#         # too many parameters, undo it
#         return z
#     else:
#         return -x**2 - y**2
#         #  (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)
#
#
# x,y=6, 4.5
# print(func(6, 4.5, 1))

a=-1
print(type(a))
