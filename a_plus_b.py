# script for testing my own python script read by wooey
import argparse
import sys

# parser and arguments
parser = argparse.ArgumentParser(description='a plus b problem')
parser.add_argument('--a', help='first num', type=int, default=0)
parser.add_argument('--b', help='second num', type=int, default=0)


def main():
    args = parser.parse_args()
    a = args.a
    b = args.b
    sum = a + b
    print('{0} + {1} = {2}'.format(a, b, sum))
    return 0


if __name__ == "__main__":
    sys.exit(main())
    #main()