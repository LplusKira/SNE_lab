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
    f = open(featureFile)
    occupationDict = getOccupationDict()
    u2f = {}
    for l in f:
        # Spec in ml-100k data's readme
        l = l.strip().split('::')
        usr = l[0] 
        gender = encodeGender(l[1])
        age = encodeAge(l[2])
        occupation = encodeOccupation(l[3], occupationDict)
        u2f[usr] = ','.join([usr, age, gender, occupation])
    f.close()
    
    # Write each user's one-hot encoded features
    f = open(outFile, 'w')
    for u in u2f:
        f.write(u2f[u] + '\n')
    f.close()

if __name__ == '__main__':
    main(sys.argv[1:3])
