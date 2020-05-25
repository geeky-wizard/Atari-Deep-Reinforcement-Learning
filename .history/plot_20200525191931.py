from glob import glob
import matplotlib.pyplot as plt


def plot_score(moving_avg, name=" "):
    x = []
    y = []
    plt.figure(dpi=300)
    if '-' in name:
        title = name.split('-')[0]
    elif '_' in name:
        title = name.split('_')[0]
    elif ' ' in name:
        title = name.split(' ')[0]
    else:
        title=name
    plt.title(title)
    [(x.append(i), y.append(j)) for i,j in enumerate(moving_avg)]
    plt.scatter(x,y,marker = '|', c=[-i for i in y])
    plt.ylabel(" Score")
    plt.xlabel("Number of Games Played")
    plt.show()
    #plt.savefig(name+" Score.png", transparent=True)
    plt.close()


def plot_avg(moving_avg, name=" ", n=10):
    x = []
    y = []
    sum_ = 0
    for i,j in enumerate(moving_avg):
        sum_ += j
        if i%n==0:
            x.append(i)
            y.append(sum_/n)
            sum_ = 0
    plt.figure(dpi=300)
    if '-' in name:
        title = name.split('-')[0]
    elif '_' in name:
        title = name.split('_')[0]
    elif ' ' in name:
        title = name.split(' ')[0]
    else:
        title=name
    plt.title(title)
    plt.plot(x,y, '-')
    plt.ylabel("Average over {:d} Episodes".format(n))
    plt.xlabel("Number of Games Played")
    plt.show()
    plt.savefig("Assets/Weights/"+name+" Avg over"+str(n)+".png", transparent=True)
    plt.close()

    
def main(file = None):
    if file is None:
        file = glob("log*.txt")
        file = file[int(input(str([x for x in zip(range(1, file.__len__()+1), file)])[2:-2].replace("), (", "\n").replace(',', ':')+'\n\nEnter File Number: '))-1]
    avg = []
    for line in open(file, 'r').readlines(-1):
        try:
            avg.append(float((line.rsplit('\t',4)[1])[14:]))
        except:
            pass
    print('\nNo. of Episodes = ', avg.__len__(), '\nAverage Score = {:0.2f}'.format(sum(avg)/avg.__len__()))
    
    while True:
        c = int(input('\n\n1. Plot Score vs Episode\n2. Plot Avgerage over n Episodes\n0. Exit\n:'))
        
        if c==1:
            plot_score(avg, file[4:-4])
        elif c==2:
            plot_avg(avg, file[4:-4], int(input('\nEnter number of Episodes to average: ')))
        elif c==0:
            return None
                
        
if __name__ == '__main__':
    main()
