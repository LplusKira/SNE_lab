import sys


def main(argv):
    featureFile = argv[0]
    uniqUsrsFile = argv[1]
    outFile = argv[2]

    # Load uniq usrs
    # - raise IOError, let SIGKILL/Others be it
    allUsrs = []
    try:
        with open(uniqUsrsFile, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                allUsrs.append(line[0])
    except IOError:
        raise

    # Save u2f
    # - raise IOError, let SIGKILL/Others be it
    u2f = {}
    try:
        with open(featureFile, 'r') as f:
            for lineNum, line in enumerate(f):
                line = line.strip().split('\t')
                usrs = line[1:]
                for u in usrs:
                    u2f[u] = u2f.get(u, []) + [lineNum]
    except IOError:
        raise
    else:
        totalAttrsNum = lineNum + 1  # Since lineNum starts at 0

    # Write each user's one-hot encoded features
    # - raise IOError, let SIGKILL/Others be it
    hasAttr, noAttr = '1,0', '0,1'
    attrInds = range(totalAttrsNum)
    try:
        with open(outFile, 'w') as f:
            for u in allUsrs:
                usrAttrInds = u2f.get(u, [])
                codes = [hasAttr if ind in usrAttrInds else noAttr for ind in attrInds]
                f.write(u + ',' + ','.join(codes) + '\n')
    except IOError:
        raise


if __name__ == '__main__':
    main(sys.argv[1:4])
