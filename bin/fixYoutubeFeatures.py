import sys

def main(argv):
    featureFile = argv[0]
    outFile = argv[1]
    # Save u2f
    f = open(featureFile)
    cnt = 0
    u2f = {}
    for l in f:
        cnt += 1
        try:
            usrs = l.strip().split('\t')
            for u in usrs:
                u2f[u] = u2f.append(cnt) if u in u2f else [cnt]
        except:
            # TODO: handle
            pass
    f.close()
    
    # Write each user's one-hot encoded features
    f = open(outFile, 'w')
    BASE = ['0,1'] * cnt
    for u in u2f:
        code = BASE[:]
        for ind in u2f[u]:
            code[ind - 1] = '1,0'
        f.write(u + ',' + ','.join(code) + '\n')
    f.close()

if __name__ == '__main__':
    main(sys.argv[1:3])
