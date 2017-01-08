import dataloaders.movielens100k as dataloader_movielens100k

def main():
    # load data as {
    #   usr_i: [ item_i0, item_i1, ... ]
    #   ... 
    # }
    dataloader = dataloader_movielens100k.dataloader_movielens100k()
    usr2items = dataloader.load('data/u.template.BOI')
    print 'qq', usr2items

if __name__ == '__main__':
    main()
