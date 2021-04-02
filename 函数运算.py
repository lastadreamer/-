import numpy as np
import matplotlib.pyplot as plt



def strucfunction(x,y): ###构造函数
    fun_dict = {}
    for i in range(len(x)):
        fun_dict[x[i]] = y[i]
    return fun_dict

def multiplicate_fun(F1,F2): ###函数与函数或与实数相乘
    F3 = {}
    if type(F2) == int or type(F2) == float:
        for i in F1:
            F1[i] = F1[i] * F2
        return F1
    else:
        if list(F1.keys())[0] > list(F2.keys())[0]:
            temp = F1
            F1 = F2
            F2 = temp
        for i in F1.keys():
            if i in F2:
                F3[i] = F1[i]*F2[i]
            else:
                F3[i] = 0
        for i in F2.keys():
            if i not in F3:
                if i in F1:
                    F3[i] = F1[i]*F2[i]
                else:
                    F3[i] = 0
        return F3
    
def segmentate_fun(X,Y): ###构造分段函数
    Fun = {}
    for i in range(len(X)):
        for j in range(len(X[i])):
            Fun[X[i][j]] = Y[i][j]
    return Fun

def add_fun(F1,F2): ###函数相加
    F3 = {}
    x1 = list(F1.keys())
    x2 = list(F2.keys())
    if x1[0] > x2[0]:
        temp = F1
        F1 = F2
        F2 = temp
    for x in F1:
        try:
            F3[x] = F1[x] + F2[x]
        except KeyError:
            F2[x] = 0
            F3[x] = F1[x] + F2[x]
    for x in F2:
        if x not in F3:
            try:
                F3[x] = F1[x] + F2[x]
            except KeyError:
                F1[x] = 0
                F3[x] = F1[x] + F2[x]
    return F3

def integrate(F1,x):
    f_sum = 0
    for i in range(len(x)-1):
        f_sum += (x[i+1]-x[i]) * F1[x[i]]
    return f_sum
    
class Frequency_domain: ###傅里叶变换，频域图
    def __init__(self):
        self.n = None
        self.T = None
        self.an = []
        self.bn = []
        self.dt = None

    def fourier_trf(self,F1,n):
        self.n = n #谐波阶数
        x1 = list(F1.keys()) 
        dt = x1[1] - x1[0] #采样频率
        self.dt = dt
        N = len(x1) #点数
        T = N-1 #间隔数
        L = T*dt
        self.T = T
        a0 = 0
        an = []
        bn = []
        f1_f = []
        for i in range(N):
            a0 += F1[x1[i]] * dt
        an.append(a0/L)
        bn.append(0)
        for j in range(1,n+1):
            an.append(0)
            bn.append(0)
            for i in range(N):
                an[j] += (2/L * F1[x1[i]] * np.cos(j*2*np.pi*i/T) * dt)
                bn[j] += (2/L * F1[x1[i]] * np.sin(j*2*np.pi*i/T) * dt)
        self.an = an
        self.bn = bn
        for i in range(N):
            f1_f.append(0)
            for j in range(n+1):
                f1_f[i] += (an[j] * np.cos(j*2*np.pi*i/T) + bn[j] * np.sin(j*2*np.pi*i/T))
        return strucfunction(x1,f1_f)
    def plot_fredom(self):
        amp = []
        fre = []
        for i in range(self.n+1):
            amp.append(np.sqrt(self.an[i]**2 + self.bn[i]**2))
            fre.append(i/(self.T * self.dt))
        df = 1/(self.T * self.dt)
        jd = len(str(df))-len(str(int(df)))-1
        fre = np.round(np.array(fre),jd)
        print(fre)
        return strucfunction(fre,amp)
    
def moveFunc(F1,k):
    jd = len(str(k))-len(str(int(k)))-1
    F1_x = np.array(list(F1.keys()))
    F1_x = np.round(F1_x+k,jd)
    F1_x = list(F1_x)
    new_F1 = {}
    for i in range(len(F1_x)):
        new_F1[F1_x[i]] = list(F1.values())[i]
    return new_F1

def symmetric(F1,axis = 'x'):
    if axis == 'x':
        F1_x = np.array(list(F1.keys()))
        F1_y = np.array(list(F1.values()))
        F1_y *= -1
        new_F1 = {}
        for i in range(len(F1_x)):
            new_F1[F1_x[i]] = F1_y[i]
        return new_F1
    elif axis == 'y':
        F1_x = np.array(list(F1.keys()))
        F1_x = np.array(list(reversed(F1.keys())))
        F1_x *= -1
        F1_y = np.array(list(reversed(F1.values())))
        new_F1 = {}
        for i in range(len(F1_x)):
            new_F1[F1_x[i]] = F1_y[i]
        return new_F1

def convolute(F1,F2,dT): #无法对频域中的图像进行卷积，原因是频率间隔的不一致
    F1_y = symmetric(F1,'y')
    move_x = list(F2.keys())[0] - list(F1_y.keys())[-1]
    dt = dT
    F1_y = moveFunc(F1_y,move_x)
    con_sum = 0
    for i in F1_y:
        if i in F2:
            con_sum += (F1_y[i] * F2[i] * dt)
    x=[]
    y=[]
    x.append(move_x)
    y.append(con_sum)
    n = 1
    while list(F1_y.keys())[0] <= list(F2.keys())[-1]:
        F1_y = moveFunc(F1_y,dt)
        con_sum = 0
        for i in F1_y:
            if i in F2:
                con_sum += (F1_y[i] * F2[i] * dt)
        x.append((n*dt+move_x))
        y.append(con_sum)
        n += 1
    return strucfunction(x,y)