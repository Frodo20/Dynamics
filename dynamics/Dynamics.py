import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation  
import imageio
from PIL import Image, ImageSequence
from kinematics.Constraints import *
from kinematics.Drives import *

class Dynamics:
    def __init__(self,paras,M=np.zeros(1),Q=np.zeros(1)):
        self.T = 0
        self.Y = 0
        self.Lambda = 0
        self.M=M
        self.Q=Q
        self.all_constraints = []
        for para in paras:
            '''
            para:[type, j, S_j, v_j, i, S_i, v_i, c, c_bar, C]
            '''
            type = para[0]
            
            #读取表格参数,创建约束矩阵phi
            if type =='ax':
                ax1 = ax(i=para[4],S_i=para[5],c=para[7])
                self.all_constraints.append(ax1)
            elif type == 'ay':
                ay1 = ay(i=para[4],S_i=para[5],c=para[7])
                self.all_constraints.append(ay1)
            elif type == 'aphi':
                aphi1 = aphi(i=para[4],c=para[7])
                self.all_constraints.append(aphi1)
            elif type =='ad':
                ad1 = ad(i=para[4],S_i=para[5],c=para[7],C=para[9])
                self.all_constraints.append(ad1)
            elif type == 'rx':
                rx1 = rx(i=para[4],S_i=para[5],j=para[1],S_j=para[2],c=para[7])
                self.all_constraints.append(rx1)
            elif type =='ry':
                ry1 = ry(i=para[4],S_i=para[5],j=para[1],S_j=para[2],c=para[7])
                self.all_constraints.append(ry1)
            elif type == 'rphi':
                rphi1 = rphi(i=para[4],j=para[1],c=para[7])
                self.all_constraints.append(rphi1)
            elif type=='rd':
                rd1 = rd(i=para[4],S_i=para[5],j=para[1],S_j=para[2],c=para[7])
                self.all_constraints.append(rd1)
            elif type=='r':
                r1 = r(i=para[4],S_i=para[5],j=para[1],S_j=para[2])
                self.all_constraints.append(r1)
            elif type=='t':
                t1 = t(i=para[4],S_i=para[5],v_i=para[6],j=para[1],S_j=para[2],v_j=para[3])
                self.all_constraints.append(t1)
            elif type=='rt':
                rt1 = t(i=para[4],S_i=para[5],v_i=para[6],j=para[1],S_j=para[2],c=para[7])
                self.all_constraints.append(rt1)

            elif type =='axd':
                axd1 = ax(i=para[4],S_i=para[5],c_bar=para[8])
                self.all_constraints.append(axd1)
            elif type == 'ayd':
                ayd1 = ay(i=para[4],S_i=para[5],c_bar=para[8])
                self.all_constraints.append(ayd1)
            elif type =='aphid':
                aphid1 = aphid(i=para[4],c_bar=para[8])
                self.all_constraints.append(aphid1)
            elif type =='add':
                add1 = ad(i=para[4],S_i=para[5],c=para[7],C=para[9],c_bar=para[8])
                self.all_constraints.append(add1)
            elif type == 'rphid':
                rphid1 = rphid(i=para[4],j=para[1],c_bar=para[8])
                self.all_constraints.append(rphid1)
            elif type=='rdd':
                rdd1 = rd(i=para[4],S_i=para[5],j=para[1],S_j=para[2],c_bar=para[8])
                self.all_constraints.append(rdd1)
            elif type=='tdd': #i、j互换
                tdd1 = tdd(i=para[1],S_i=para[2],v_i=para[3],j=para[4],S_j=para[5])
                self.all_constraints.append(tdd1)
            elif type=='tdd8': #i、j互换
                tdd1 = tdd8(i=para[1],S_i=para[2],v_i=para[3],j=para[4],S_j=para[5])
                self.all_constraints.append(tdd1)
            elif type=='tdd12': #i、j互换
                tdd1 = tdd12(i=para[1],S_i=para[2],v_i=para[3],j=para[4],S_j=para[5])
                self.all_constraints.append(tdd1)
                
            elif type=='c': #专用于求解平衡问题的接触约束
                c1 = c(i=para[4],S_i=para[5],v_i=para[6],c=para[7])
                self.all_constraints.append(c1)
    
    #构建约束矩阵            
    def constraints(self,q,t):
        i = 1
        for constraint in self.all_constraints:
            if i == 1:
                phi = constraint.constraints(q,t)
            else:
                phi = np.vstack((phi,constraint.constraints(q,t)))
            #print(i)
            #print(phi)
            i+=1
        #print (i)       
        return phi
    
    #构建雅克比矩阵
    def Jacobi(self,q,t,l):
        i =1
        for constraint in self.all_constraints:
            if i == 1:
                J = constraint.Jacobi(q,t,l)
                #print(J)
            else:
                #print(constraint.Jacobi(q,t,l))
                J = np.vstack((J,constraint.Jacobi(q,t,l)))
            #print(i)
            #print(constraint.Jacobi(q,t,l))
            #print(J)
            i+=1            
        return J

    #构建速度方程右项            
    def v(self,q,t):
        i=1
        for constraint in self.all_constraints:
            if i==1:
                v = constraint.v(q, t)
            else:
                v = np.vstack((v,constraint.v(q, t)))
            #print(i)
            #print(constraint.v(q, t))
            i+=1
        #print(v)
        return v

    #构建加速度右项        
    def gamma(self,q,dq,t):
        i=1
        for constraint in self.all_constraints:
            #print(constraint.gamma(q, dq, t))
            if i==1:
                gamma = constraint.gamma(q,dq,t)
                #print (gamma)
            else:
                gamma = np.vstack((gamma,constraint.gamma(q,dq,t)))
            #print(i)
            #print(gamma)
            i+=1
                
        return gamma
    

    def fun(self, t=0, y=np.array([[0]])):
    
        q = y[0:len(y)//2]
        dq = y[len(y)//2:len(y)]
        l = len(q)
        #print(l)
        
        Jacobi = self.Jacobi(q, t, l)
        #print(Jacobi[7,:])
        gamma = self.gamma(q, dq, t)
        #print(gamma)
        
        z = np.zeros((len(Jacobi),len(Jacobi)))
        
        A1 = np.concatenate((self.M,Jacobi.T),axis=1)
        A2 = np.concatenate((Jacobi,z),axis=1)
        A = np.concatenate((A1,A2),axis=0)

        Q = (self.Q).copy()
        
        #print(y[15])
        #print(y[6])
        if y[15]>0 and y[6]<1.5:
            pass
        
        elif y[15]>0 and y[6]>=1.5 and y[6]<=5:
            Q[6]=Q[6]+282857/(y[6]-6)+62857
            
        elif y[15]>0 and y[6]>5 and y[6]<=5.5:
            Q[6]=Q[6]-110000*(1-np.sin(2*np.pi*(y[6]-5.25)))
            
        else:
            pass
        
        #print(Q)
            
        B = np.concatenate((Q,gamma),axis=0)
        
        X = np.dot(np.linalg.inv(A),B)
        
        X1 = X[0:len(y)//2]
        X2 = X[len(y)//2:len(y)//2+len(Jacobi)]
        
        dy = np.concatenate((dq,X1),axis=0)
        lambda0 = X2
        
        return dy, lambda0

    def RK(self, t=0, y=np.array([[0]]), dt=0.01):
        k=[]
        y0=y
        for i in range(4):
            if i==0:
                dy,lambda0 = self.fun(t,y0)

                #print(dy)
                #print(lambda0)
            elif i==1 or i==2:
                dy,lambda0 = self.fun(t+dt/2,y0+k[i-1]/2)
                ''' if i==1:
                    print(dy)
                    print(lambda0) '''     
            else:
                dy,lambda0 = self.fun(t+dt,y0+k[i-1])
            k.append(dt*dy)
            
        #print(k[0])
        #print(type(k[0]))        
        y = y0+1/6*(k[0]+2*k[1]+2*k[2]+k[3])
        #print(type(y))
        return y

    def correct(self,y=np.array([[0]]),t=0,eps=1e-6):
        
        q = y[0:len(y)//2]
        dq = y[len(y)//2:len(y)]
        l=len(q)
        
        phi = self.constraints(q, t)
        Jacobi = self.Jacobi(q, t, l)
        #print(self.constraints(q, t))
        #print(self.all_constraints)
        phid = np.dot(Jacobi,dq)-self.v(q, t)
        H = np.dot(Jacobi.T,np.linalg.inv(np.dot(Jacobi,Jacobi.T)))
        
        if np.linalg.norm(phi)>eps:
            q = q-np.dot(H,phi)
        
        if np.linalg.norm(phid)>eps:
            dq = dq-np.dot(H,phid)
       #print(len(q))
        #print(len(dq))
        y = np.concatenate((q,dq),axis=0)
        
        return y 
        
    def solve1(self,q0=np.array([[0]]),dq0=np.array([[0]]),t0=0,te=1.5,num=500,eps=1e-6,iter_max=15):
        #设定初值
        q = q0
        dq = dq0
        l = len(q)
        Y = np.zeros((num+1,3*l))
        y = np.concatenate((q,dq),axis=0)
        
        dt = (te-t0)/num
        #print(dt)
        T = np.arange(t0,te+dt,dt)
        
        #phi = self.constraints(q, t)
        
        Lambda = np.zeros((num+1,len(self.Jacobi(q, t0, l))))
        #print(Lambda.shape)
        t=t0
        for i in range(num+1):
            #print(i)
            if i==0:
                #print('i=0')
                dy, lambda0 = self.fun(t, y)
                #print((np.concatenate((y,dy[l:2*l]),axis=0)).T)
                Y[i,:] = (np.concatenate((y,dy[l:2*l]),axis=0)).T
                #print(lambda0.shape)
                Lambda[i,:] = lambda0.T
                #print(Y[0,:])
            else:   
                #print('i=')
                #print(i)
                
                y = self.RK(t, y, dt)
                ''' if i==1:
                    print(y) '''
                
                y = self.correct(y, t,eps)
                
                #print(len(y))
                t += dt
                
                dy, lambda0=self.fun(t, y)
                Y[i,:] = (np.concatenate((y,dy[l:2*l]),axis=0)).T
                Lambda[i,:] = lambda0.T
        
        self.Y=Y
        self.T=T   
        self.Lambda=Lambda
        return Y, Lambda
    
    def solve2(self,q0=np.array([[0]]), t0=0, te=5, num=500, eps1=1e-10, eps2=1e-10, iter_max=15):
        #设定初值
        q = q0
        l = len(q)
        #运动学输出结果
        Y = np.zeros((num+1,3*l)) 

        
        dt = (te-t0)/num
        T = np.arange(t0,te+dt,dt)
        
        Lambda = np.zeros((num+1,len(self.Jacobi(q, t0, l))))
        
        for i in range(num+1):
            t = t0+i*dt
            phi = self.constraints(q, t)
            #print(phi)   
            delta1 = np.linalg.norm(phi)
            iter_num = 0
            
            while delta1 > eps1:
                Jacobi = self.Jacobi(q, t, l)
                #print(Jacobi)
                if abs(np.linalg.det(Jacobi))<eps2:
                    print('Improper initial value,1')
                    
                
                dq = -np.dot(np.linalg.inv(Jacobi),phi)
                q = q+dq
                phi = self.constraints(q, t)
                delta1 = np.linalg.norm(phi)
                
                iter_num +=1
                if iter_num > iter_max:
                    print('Improper initial value,2')
                    return 
                
            Jacobi = self.Jacobi(q,t,l)
            v =self.v(q, t)
            if np.abs(np.linalg.det(Jacobi))<eps2:
                print('Sigular configuration')
                return
            
            dq = np.dot(np.linalg.inv(Jacobi),v)
            gamma = self.gamma(q, dq, t)
            ddq = np.dot(np.linalg.inv(Jacobi),gamma)
            Y[i,...] = np.c_[q.T,dq.T,ddq.T] #Y=[q,dq,ddq]
            
            lambda0 = np.dot(np.linalg.inv(Jacobi.T),self.Q-np.dot(self.M,ddq))
            
            Lambda[i,:] = lambda0.T
            
            q = q+dq*dt+ddq*dt**2/2
        
        self.Y=Y
        #print(Y[:,2])
        #print(Y[:,5])
        self.T=T
        self.Lambda=Lambda
        #print(Y[100,:])
        return Y,Lambda
    
    def solve3(self,q0=np.array([[0]]), eps=1e-5, iter_max=15):
        q=q0
        l=len(q)
        #print(l)
        l1=4#杆长
        
        Qu=np.array([[0],[-9.8]])
        Qv=np.array([[0]])
        
        
        Jacobi = self.Jacobi(q, 0, l)
            
        Ju = Jacobi[:,0:2]
        Jv = np.array([[Jacobi[0,2]],[Jacobi[1,2]]])
        
        H = np.dot(np.linalg.inv(Ju),Jv)
        
        EC = np.dot(H.T,Qu)-Qv
        #print(EC)
        
        C1 = np.concatenate((EC,self.constraints(q0, t)))
        
        iter_num=0
        while np.linalg.norm(C1)>eps:
            assert iter_num<=iter_max, 'Improper initial position'
            #求雅可比
            #print(q[2,0])
            print(q)
            J1 = 9.8*np.array([[1,np.tan(q[2,0]),q[1,0]/np.cos(q[2,0])**2-l1/2*(np.sin(q[2,0])+np.tan(q[2,0])/np.cos(q[2,0]))]])
            J = np.concatenate((J1,Jacobi),axis=0)
            
            #更新位姿
            q = q - np.dot(np.linalg.inv(J),C1)
            
            #更新约束阵
            Jacobi = self.Jacobi(q, 0, l)
            Ju = Jacobi[:,0:2]
            Jv = np.array([[Jacobi[0,2]],[Jacobi[1,2]]])
            
            H = np.dot(np.linalg.inv(Ju),Jv)
            
            EC = np.dot(H.T,Qu)-Qv
            
            C1 = np.concatenate((EC,self.constraints(q, t)))
            
            iter_num+=1
            
        lambda0 = np.dot((np.linalg.inv(Ju)).T,Qu)
            
        return q, lambda0
    
    #绘制运动图像
    def plot1(self,plot_phi=0,plot_dphi=0,plot_x=0,plot_dx=0,plot_Fx=1,plot_Fy=1):
        font = {'family': 'Times New Roman',
        'color':  'darkred',
        'weight': '900',
        'size': 13,
        }
        
        #设置图的样式
        plt.style.use('seaborn-paper')
        #设置画布大小
        plt.figure(figsize=(10, 4),dpi=120)
        
        if plot_phi==1:
            plt.subplot(1,2,1)
            #print(self.Y[...,2])
            plt.plot(self.T, self.Y[...,2])
            
            plt.grid()
            plt.title("$\\theta_1-t$",fontdict=font)
            plt.xlabel("$t(s)$",fontsize=8,loc='right')
            plt.ylabel("$\\theta_1(rad)$",fontsize=8)
            
        if plot_dphi==1:
            plt.subplot(1,2,2)
            #print(self.Y[...,2])
            plt.plot(self.T, self.Y[...,11])
            
            plt.grid()
            plt.title("$w_1-t$",fontdict=font)
            plt.xlabel("$t(s)$",fontsize=8,loc='right')
            plt.ylabel("$w_1(rad/s)$",fontsize=8) 
        
        if plot_x==1:
            plt.subplot(1,2,1)
            #print(self.Y[...,2])
            plt.plot(self.T, self.Y[...,6])
            
            plt.grid()
            plt.title("$x_3-t$",fontdict=font)
            plt.xlabel("$t(s)$",fontsize=8,loc='right')
            plt.ylabel("$x_3(m)$",fontsize=8)
        
        if plot_dx==1:
            plt.subplot(1,2,2)
            #print(self.Y[...,2])
            plt.plot(self.T, self.Y[...,15])
            
            plt.grid()
            plt.title("$\dot{x_3}-t$",fontdict=font)
            plt.xlabel("$t(s)$",fontsize=8,loc='right')
            plt.ylabel("$\dot{x_3}(m/s)$",fontsize=8)
            
        if plot_Fx==1:
            plt.subplot(1,2,1)
            #print(self.Y[...,2])
            plt.plot(self.T, -self.Lambda[...,0])
            
            plt.grid()
            plt.title("$F_x-t$",fontdict=font)
            plt.xlabel("$t(s)$",fontsize=8,loc='right')
            plt.ylabel("$F_x(N)$",fontsize=8)
            
        if plot_Fy==1:
            plt.subplot(1,2,2)
            #print(self.Y[...,2])
            plt.plot(self.T, -self.Lambda[...,1])
            
            plt.grid()
            plt.title("$F_y-t$",fontdict=font)
            plt.xlabel("$t(s)$",fontsize=8,loc='right')
            plt.ylabel("$F_y(N)$",fontsize=8)      
            
          
        plt.subplots_adjust(wspace=0.4,hspace=0.5)
        plt.show()
        
        
    #绘制运动图像
    def plot2(self,plot_Fx=0,plot_Fy=0,plot_M1=1,plot_M2=1):
        font = {'family': 'Times New Roman',
        'color':  'darkred',
        'weight': '900',
        'size': 13,
        }
        
        #设置图的样式
        plt.style.use('seaborn-paper')
        #设置画布大小
        plt.figure(figsize=(10, 4),dpi=120)
        
        if plot_Fx==1:
            plt.subplot(1,2,1)
            #print(self.Y[...,2])
            plt.plot(self.T, -self.Lambda[...,0])
            
            plt.grid()
            plt.title("$F_{1x}-t$",fontdict=font)
            plt.xlabel("$t(s)$",fontsize=8,loc='right')
            plt.ylabel("$F_{1x}(N)$",fontsize=8)
            
        if plot_Fy==1:
            plt.subplot(1,2,2)
            #print(self.Y[...,2])
            plt.plot(self.T, -self.Lambda[...,1])
            
            plt.grid()
            plt.title("$F_{1y}-t$",fontdict=font)
            plt.xlabel("$t(s)$",fontsize=8,loc='right')
            plt.ylabel("$F_{1y}(N)$",fontsize=8) 
        
        if plot_M1==1:
            plt.subplot(1,2,1)
            #print(self.Y[...,2])
            plt.plot(self.T, -self.Lambda[...,-2])
            
            plt.grid()
            plt.title("$M_1-t$",fontdict=font)
            plt.xlabel("$t(s)$",fontsize=8,loc='right')
            plt.ylabel("$M_1(N*m)$",fontsize=8)
        
        if plot_M2==1:
            plt.subplot(1,2,2)
            #print(self.Y[...,2])
            plt.plot(self.T, -self.Lambda[...,-1])
            
            plt.grid()
            plt.title("$M_2-t$",fontdict=font)
            plt.xlabel("$t(s)$",fontsize=8,loc='right')
            plt.ylabel("$M_2(N*m)$",fontsize=8)   
            
          
        plt.subplots_adjust(wspace=0.4,hspace=0.5)
        plt.show()
            
            