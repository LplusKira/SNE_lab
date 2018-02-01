import sys
def encodeAge(s):
    age = int(s)
    if age <= 17:
        return '1,0,0,0'
    elif age <= 35:
        return '0,1,0,0'
    elif age <= 65:
        return '0,0,1,0'
    else:
        return '0,0,0,1'

def encodeGender(c):
    return '1,0' if c == 'M' else '0,1'

def getOccupationDict():
    occupation2order = {
        'lawyer': 10,
        'executive': 7,
        'programmer': 15,
        'retired': 16,
        'administrator': 1,
        'salesman': 17,
        'student': 19,
        'doctor': 3,
        'engineer': 5,
        'other': 14,
        'librarian': 11,
        'healthcare': 8,
        'marketing': 12,
        'artist': 2,
        'scientist': 18,
        'educator': 4,
        'writer': 21,
        'none': 13,
        'entertainment': 6,
        'homemaker': 9,
        'technician': 20,
    }
    size = len(occupation2order)
    return {k: ((v - 1) * '0,' + '1,' + (size - v) * '0,').strip(',') for k, v in occupation2order.iteritems()}

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
        try:
            # Spec in ml-100k data's readme
            l = l.strip().split('|')
            usr = l[0] 
            age = encodeAge(l[1])
            gender = encodeGender(l[2])
            occupation = encodeOccupation(l[3], occupationDict)
            u2f[usr] = ','.join([usr, age, gender, occupation])
        except:
            # TODO: handle
            pass
    f.close()
    
    # Write each user's one-hot encoded features
    f = open(outFile, 'w')
    for u in u2f:
        f.write(u2f[u] + '\n')
    f.close()

if __name__ == '__main__':
    main(sys.argv[1:3])
