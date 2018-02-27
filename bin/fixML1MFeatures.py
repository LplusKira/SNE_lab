import sys


def encodeAge(s):
    age = s
    if age == '1':
        return '1,0,0,0,0,0,0'
    elif age == '18':
        return '0,1,0,0,0,0,0'
    elif age == '25':
        return '0,0,1,0,0,0,0'
    elif age == '35':
        return '0,0,0,1,0,0,0'
    elif age == '45':
        return '0,0,0,0,1,0,0'
    elif age == '50':
        return '0,0,0,1,0,1,0'
    else:
        return '0,0,0,0,0,0,1'


def encodeGender(c):
    return '1,0' if c == 'M' else '0,1'


def getOccupationDict():
    size = 21
    return {str(v): (v * '0,' + '1,' + (size - v - 1) * '0,').strip(',') for v in range(size)}


def encodeOccupation(s, d):
    return d[s]


def main(argv):
    featureFile = argv[0]
    outFile = argv[1]

    # Save u2f
    occupationDict = getOccupationDict()
    u2f = {}
    try:
        with open(featureFile, 'r') as f:
            for line in f:
                # Ref: ml-1m data's readme
                line = line.strip().split('::')
                usr = line[0]
                gender = encodeGender(line[1])
                age = encodeAge(line[2])
                occupation = encodeOccupation(line[3], occupationDict)
                u2f[usr] = ','.join([usr, age, gender, occupation])
    except IOError:
        raise

    # Write each user's one-hot encoded features
    with open(outFile, 'w') as f:
        for u in u2f:
            f.write(u2f[u] + '\n')


if __name__ == '__main__':
    main(sys.argv[1:3])
