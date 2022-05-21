import numpy as np
import matplotlib.pyplot as plt
from dynamics.Dynamics import Dynamics
from dynamics.Animate import *

if __name__=="__main__":
    
    ''' paras1=[\
    ['ax', 0, np.zeros((2,1)), np.zeros((2,1)), 1, np.array([[-1],[0]]), np.zeros((2,1)), 0, 0], \
    ['ay', 0, np.zeros((2,1)), np.zeros((2, 1)), 1, np.array([[-1],[0]]), np.zeros((2, 1)), 0, 0], \
    ['r', 2, np.array([[-1.75],[0]]), np.zeros((2, 1)), 1, np.array([[1],[0]]), np.zeros((2, 1)), 0, 0], \
    ['r', 3, np.zeros((2,1)), np.zeros((2,1)), 2, np.array([[1.75],[0]]), np.zeros((2,1)), 0, 0], \
    ['aphi',0,np.zeros((2,1)),np.zeros((2,1)),3,np.zeros((2,1)),np.zeros((2,1)), 0, 0], \
    ['ay', 0, np.zeros((2,1)), np.zeros((2,1)), 3, np.zeros((2,1)), np.zeros((2,1)), 0, 0] \
    ]
    
    M1 = np.diag([200,200,450,35,35,35,25,25,0.02])
    
    Q1 = np.array([[1960],[0],[41450],[343],[0],[0],[245],[0],[0]]) #主动力较复杂，定义在类中
    
    D1 = Dynamics(paras1, M1, Q1)
    
    q01 = np.array([[-1],[0],[np.pi],[-0.25],[0],[0],[1.5],[0],[0]])
    
    dq01 = np.array([[0],[-30],[30],[0],[-30],[120/7],[0],[0],[0]])
    
    dq02 = np.array([[0],[-60],[60],[0],[-60],[240/7],[0],[0],[0]])
    
    Y1, lambda01 = D1.solve1(q01,dq01)
    
    Y2, lambda02 = D1.solve1(q01,dq02)
    
    D1.plot1() '''
    
    ''' q1 = Y1[:,0:9]
    
    Animate1(q1,1) '''
    
    ''' T = np.arange(0,1.503,0.003)
    #设置图的样式
    plt.style.use('seaborn-paper')
    #设置画布大小
    plt.figure(figsize=(12, 4),dpi=120)
    plt.subplot(1,2,1)
    plt.plot(T, -lambda01[...,0],linewidth=1.75,label='$w_{11}(0)$')
    plt.plot(T, -lambda02[...,0],linewidth=1.75,label='$w_{12}(0)$')
    #plt.ylim(-1,5)
    plt.xlabel("$t(s)$",fontsize=10,loc='right')
    plt.ylabel("$F_{1x}(N)$",fontsize=10)
    plt.grid()
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(T, -lambda01[...,1],linewidth=1.75,label='$w_{11}(0)$')
    plt.plot(T, -lambda02[...,1],linewidth=1.75,label='$w_{12}(0)$')
    #plt.ylim(-1,5)
    plt.xlabel("$t(s)$",fontsize=10,loc='right')
    plt.ylabel("$F_{1y}(N)$",fontsize=10)
    plt.grid()
    plt.legend()
    plt.subplots_adjust(wspace=0.4,hspace=0.5)
    plt.show() '''
    
    ''' paras2=[\
    ['ax', 0, np.zeros((2,1)), np.zeros((2,1)), 1, np.array([[-0.5],[0]]), np.zeros((2,1)), 0, 0], \
    ['ay', 0, np.zeros((2,1)), np.zeros((2, 1)), 1, np.array([[-0.5],[0]]), np.zeros((2, 1)), 0, 0], \
    ['r', 2, np.array([[-0.5],[0]]), np.zeros((2, 1)), 1, np.array([[0.5],[0]]), np.zeros((2, 1)), 0, 0], \
    ['aphid', 0, np.zeros((2,1)), np.zeros((2,1)), 1, np.zeros((2,1)), np.zeros((2,1)), 0, 0], \
    ['rphid', 2, np.zeros((2,1)), np.zeros((2,1)), 1, np.zeros((2,1)), np.zeros((2,1)), 0, 0] \
    ]
    
    M2 = np.diag([1, 1, 1/12, 1, 1, 1/12])
    
    Q2 = np.array([[0],[-9.8],[0],[0],[-9.8],[0]])
    
    D2 = Dynamics(paras2, M2, Q2)
    
    q02 = np.array([[0.5], [0], [0], [1.5], [0], [0]])
    
    Y2, lambda02 = D2.solve2(q02)
    
    D2.plot2()
    
    q2 = Y2[:,0:9]
    
    Animate2(q2,1) '''
    
    paras3=[\
    ['ax', 0, np.zeros((2,1)), np.zeros((2,1)), 1, np.array([[-2],[0]]), np.zeros((2,1)), -2, 0], \
    ['c', 0, np.zeros((2,1)), np.zeros((2,1)), 1, np.zeros((2,1)), np.array([[1],[0]]), 1, 0] \
    ]
    
    D3 = Dynamics(paras3)
    
    q03 = np.array([[0],[0],[0.1]])
    ''' q03 = np.array([[-0.12],[1.02],[0.35]]) '''
    
    q3,lambda0 = D3.solve3(q03)
    
    print(q3)
    print(lambda0)
    
    plot3(q3)
    
    
    