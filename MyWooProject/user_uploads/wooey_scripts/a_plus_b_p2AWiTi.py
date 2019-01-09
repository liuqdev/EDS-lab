# # script for testing my own python script read by wooey
# import argparse
# import sys
#
# # parser and arguments
# parser = argparse.ArgumentParser(description='a plus b problem')
# parser.add_argument('--a', help('first number'), type=int, default=1)
# parser.add_argument('--b', help('second number'), type=int, default=1)
#
#
# def main():
#     args = parser.parse_args()
#     for i, p in enumerate(args):
#         print(i, p)
#     a = args.a
#     b = args.b
#     sum = a + b
#     print('Sum =', sum)
#     return sum
#
#
# if __name__ == '__main__':
#     sys.exit(main)

import argparse
import sys

parser = argparse.ArgumentParser(description="Find the sum of all the numbers below a certain number.")
parser.add_argument('--below', help='The number to find the sum of numbers below.', type=int, default=1000)

def main():
    args = parser.parse_args()
    s = sum((i for i in range(args.below)))
    print("Sum =", s)

    with open('result.txt', 'w') as fo:
        fo.write("Sum= {0}".format(s))

    return 0

if __name__ == "__main__":
    sys.exit(main())
