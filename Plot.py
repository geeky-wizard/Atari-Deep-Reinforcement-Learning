#import pandas as pd
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
    #plt.savefig(name+" Av over"+str(n)+".png", transparent=True)
    plt.close()

    
def calcutae_moving_avg(avg):
    m_av = []
    alpha = 0.0
    first = True
    for x in avg:
        if first:
            m_av.append(alpha*x)
            first = False
        else:
            m_av.append(alpha*m_av[-1] + (1 - alpha)*x)
    plot_score(m_av)
    #pd.DataFrame(m_av, columns=['avg']).to_csv('generated.txt', index=False)
    
def main():
    file = glob("Log*.txt")
    file = file[int(input(str([x for x in zip(range(1, file.__len__()+1), file)])[2:-2].replace("), (", "\n").replace(',', ':')+'\n\nEnter File Number: '))-1]
    avg = []
    for line in open(file, 'r').readlines(-1):
        try:
            avg.append(float((line.rsplit('\t',4)[1])[14:]))
        except:
            pass
    print('\nNo. of Episodes = ', avg.__len__(), '\nAverage Score = {:0.2f}'.format(sum(avg)/avg.__len__()))
    
    #c = int(input('\n1. Plot Score vs Episode\n2. Plot Avgerage over n Episodes\n0. Exit:'))
    while True:
        c = int(input('\n\n1. Plot Score vs Episode\n2. Plot Avgerage over n Episodes\n0. Exit\n:'))
        
        if c==1:
            plot_score(avg, file[4:-4])
        elif c==2:
            plot_avg(avg, file[4:-4], int(input('\nEnter number of Episodes to average: ')))
        elif c==0:
            return None
                
        
if __name__ == '__main__':
    #csv_file = pd.read_csv("History.txt", usecols=['accuracy'])
    #plot_accuracy(csv_file.accuracy)
    #csv_file = pd.read_csv("generated.txt")
    #plot_(csv_file.avg)
    #csv_file = pd.read_csv("Avg.txt")
    #calcutae_moving_avg(csv_file.avg)
    main()
