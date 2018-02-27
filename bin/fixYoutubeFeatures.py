import sys


def main(argv):
    featureFile = argv[0]
    outFile = argv[1]

    # Save u2f
    # - raise IOError, let SIGKILL/Others be it
    u2f = {}
    try:
        with open(featureFile) as f:
            for lineNum, line in enumerate(f):
                usrs = line.strip().split('\t')
                for u in usrs:
                    u2f[u] = u2f.get(u, []) + [lineNum]
    except IOError:
        raise
    else:
        totalAttrsNum = lineNum + 1  # Since lineNum starts at 0

    # Write each user's one-hot encoded features
    # - raise IOError, let SIGKILL/Others be it
    BASE = ['0,1'] * totalAttrsNum
    try:
        with open(outFile, 'w') as f:
            for u in u2f:
                code = BASE[:]
                for ind in u2f[u]:
                    code[ind - 1] = '1,0'
                f.write(u + ',' + ','.join(code) + '\n')
    except IOError:
        raise


if __name__ == '__main__':
    main(sys.argv[1:3])
