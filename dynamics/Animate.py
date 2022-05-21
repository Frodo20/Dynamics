import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation  
import imageio
from PIL import Image, ImageSequence


def Animate1(q,save_plot=1):
        
        #字体定义
        font = {'family': 'Times New Roman',
        'color':  'Black',
        'weight': '900',
        'size': 18,
        }
        
        num = q.shape[0]
        #print(num)
        
        theta = np.linspace(0,2*np.pi,num)
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        dpi_range = np.linspace(-1,1,150)
        
        #杆1
        xdata_1 = [0, -2]
        ydata_1 = [0, 0]
        line1, = ax.plot(xdata_1, ydata_1, lw=2) #线宽lw=2
        
        #杆2
        xdata_2 = [-2, 1.5]
        ydata_2 = [0, 0]
        line2, = ax.plot(xdata_2,ydata_2,lw=2)
        
        #约束铰
        circle_line, = ax.plot([],[],lw=2)
        circle_line1, = ax.plot([],[],lw=2)
        
        
        #B3
        line3, = ax.plot([],[],lw=2)
        line4, = ax.plot([],[],lw=2)
        line5, = ax.plot([],[],lw=2)
        line6, = ax.plot([],[],lw=2)
        
        # initialization function
        def init():
            # creating an empty plot/frame
            ax.set_ylim(-3,3)
            ax.set_xlim(-4,7.5)
            del xdata_1[:]
            del ydata_1[:]
            del xdata_2[:]
            del ydata_2[:]
            ''' del xdata_3[:]
            del ydata_3[:]
            del xdata_4[:]
            del ydata_4[:] '''
            line1.set_data(xdata_1,ydata_1)
            line2.set_data(xdata_2,ydata_2)
            ''' line3.set_data(xdata_3,ydata_3)
            line4.set_data(xdata_4,ydata_4) '''
            
            return line1, line2
    
        ax.grid() #网格线
        plt.axis('equal') #横纵坐标等值
        
        def run(i):
                 
            #计算转换矩阵
            A1_i = np.array([[np.cos(q[i,2]),-np.sin(q[i,2])],[np.sin(q[i,2]),np.cos(q[i,2])]])
            A2_i = np.array([[np.cos(q[i,5]),-np.sin(q[i,5])],[np.sin(q[i,5]),np.cos(q[i,5])]])
            S11 = np.array([[-1],[0]])
            S12 = np.array([[1],[0]])
            S21 = np.array([[-1.75],[0]])
            S22 = np.array([[1.75],[0]])
            
            #根据i更新1杆的位置
            r1 = q[i,0:2]+(np.dot(A1_i,S11)).T
            r2 = q[i,0:2]+(np.dot(A1_i,S12)).T
            #print(r1)
            #print(r2)
            xdata_1 = [r1[0,0],r2[0,0]]
            #print (xdata_1)
            ydata_1 = [r1[0,1],r2[0,1]]
            #print(ydata_1)
            line1.set_data(xdata_1,ydata_1)
            
            #根据i更新2杆的位置
            r3 = q[i,3:5]+(np.dot(A2_i,S22)).T
            xdata_2 = [r2[0,0],r3[0,0]]
            #print(xdata_2)
            ydata_2 = [r2[0,1],r3[0,1]]
            #print(ydata_2)
            line2.set_data(xdata_2,ydata_2)
            
            #铰链
            circle_x = [0.1*np.cos(theta[j])+r2[0,0] for j in range(len(theta))]
            circle_y = [0.1*np.sin(theta[j])+r2[0,1] for j in range(len(theta))]
            circle_line.set_data(circle_x,circle_y)
            
            #铰链
            circle_x1 = [0.1*np.cos(theta[j])+r3[0,0] for j in range(len(theta))]
            circle_y1 = [0.1*np.sin(theta[j])+r3[0,1] for j in range(len(theta))]
            circle_line1.set_data(circle_x1,circle_y1)
            
            #B3
            #上线
            xdata_3 = [r3[0,0]-0.25,r3[0,0]+0.25]
            ydata_3 = [r3[0,1]+0.15,r3[0,1]+0.15]
            line3.set_data(xdata_3,ydata_3)
            #下线
            xdata_4 = [r3[0,0]-0.25,r3[0,0]+0.25]
            ydata_4 = [r3[0,1]-0.15,r3[0,1]-0.15]
            line4.set_data(xdata_4,ydata_4)
            
            #左线
            xdata_5 = [r3[0,0]-0.25,r3[0,0]-0.25]
            ydata_5 = [r3[0,1]+0.15,r3[0,1]-0.15]
            line5.set_data(xdata_5,ydata_5)
            #右线
            xdata_6 = [r3[0,0]+0.25,r3[0,0]+0.25]
            ydata_6 = [r3[0,1]+0.15,r3[0,1]-0.15]
            line6.set_data(xdata_6,ydata_6)
            
            
        #(0,0)处的铰
        ax.plot(dpi_range/10,np.sqrt(1-dpi_range**2)/10,'b',lw=2)
        ax.plot(dpi_range/10,-np.sqrt(1-dpi_range**2)/10,'b',lw=2)
        
        
        #绘制地平线
        line7, =ax.plot([-10,10],[0,0],  'r--', lw=1)
        
        
        #绘制地面滑移铰
        #上半部分
        line8, =ax.plot([1.25,5.75],[0.15,0.15], 'b-',lw=1)
        #下半部分
        line9, =ax.plot([1.25,5.75],[-0.15,-0.15], 'b-',lw=1)
        
        
        plt.title('Animation',fontdict=font)
        
        #call the animator
        ani1 = animation.FuncAnimation(fig, run, frames=num-1, blit=False, interval=100, repeat=True, init_func=init)
            
        plt.show()

        if save_plot==1:

            ani2 = animation.FuncAnimation(fig, run, frames=num-1, blit=False, interval=100, repeat=False, init_func=init)

            ani2.save("ani1.gif",writer='imagemagick', fps=100)
            
def Animate2(q,save_plot=1):
        #字体定义
        font = {'family': 'Times New Roman',
        'color':  'Black',
        'weight': '900',
        'size': 18,
        }
        
        num = q.shape[0]
        #print(num)
        
        theta = np.linspace(0,2*np.pi,num)
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        dpi_range = np.linspace(-1,1,150)
        
        #杆1
        xdata_1 = [0, 1]
        ydata_1 = [0, 0]
        line1, = ax.plot(xdata_1, ydata_1, lw=4) #线宽lw=2
        
        #杆2
        xdata_2 = [1, 2]
        ydata_2 = [0, 0]
        line2, = ax.plot(xdata_2,ydata_2,lw=4)
        
        #约束铰
        circle_line, = ax.plot([],[],lw=2)
        
        
        # initialization function
        def init():
            # creating an empty plot/frame
            ax.set_ylim(-2,2)
            ax.set_xlim(-2,2)
            del xdata_1[:]
            del ydata_1[:]
            del xdata_2[:]
            del ydata_2[:]
            ''' del xdata_3[:]
            del ydata_3[:]
            del xdata_4[:]
            del ydata_4[:] '''
            line1.set_data(xdata_1,ydata_1)
            line2.set_data(xdata_2,ydata_2)
            ''' line3.set_data(xdata_3,ydata_3)
            line4.set_data(xdata_4,ydata_4) '''
            
            return line1, line2
    
        ax.grid() #网格线
        plt.axis('equal') #横纵坐标等值
        
        def run(i):
                 
            #计算转换矩阵
            A1_i = np.array([[np.cos(q[i,2]),-np.sin(q[i,2])],[np.sin(q[i,2]),np.cos(q[i,2])]])
            A2_i = np.array([[np.cos(q[i,5]),-np.sin(q[i,5])],[np.sin(q[i,5]),np.cos(q[i,5])]])
            S11 = np.array([[-0.5],[0]])
            S12 = np.array([[0.5],[0]])
            S21 = np.array([[-0.5],[0]])
            S22 = np.array([[0.5],[0]])
            
            #根据i更新1杆的位置
            r1 = q[i,0:2]+(np.dot(A1_i,S11)).T
            r2 = q[i,0:2]+(np.dot(A1_i,S12)).T
            #print(r1)
            #print(r2)
            xdata_1 = [r1[0,0],r2[0,0]]
            #print (xdata_1)
            ydata_1 = [r1[0,1],r2[0,1]]
            #print(ydata_1)
            line1.set_data(xdata_1,ydata_1)
            
            #根据i更新2杆的位置
            r3 = q[i,3:5]+(np.dot(A2_i,S22)).T
            xdata_2 = [r2[0,0],r3[0,0]]
            #print(xdata_2)
            ydata_2 = [r2[0,1],r3[0,1]]
            #print(ydata_2)
            line2.set_data(xdata_2,ydata_2)
            
            #铰链
            circle_x = [0.1*np.cos(theta[j])+r2[0,0] for j in range(len(theta))]
            circle_y = [0.1*np.sin(theta[j])+r2[0,1] for j in range(len(theta))]
            circle_line.set_data(circle_x,circle_y)
            
            
        #(0,0)处的铰
        ax.plot(dpi_range/10,np.sqrt(1-dpi_range**2)/10,'b',lw=2)
        ax.plot(dpi_range/10,-np.sqrt(1-dpi_range**2)/10,'b',lw=2)
        
        
        #绘制地平线
        line3, =ax.plot([-10,10],[0,0],  'r--', lw=1)
        
        
        plt.title('Animation',fontdict=font)
        
        #call the animator
        ani1 = animation.FuncAnimation(fig, run, frames=num-1, blit=False, interval=50, repeat=True, init_func=init)
            
        plt.show()

        if save_plot==1:

            ani2 = animation.FuncAnimation(fig, run, frames=num-1, blit=False, interval=50, repeat=False, init_func=init)

            ani2.save("ani2.gif",writer='imagemagick', fps=100)
    
def plot3(q,save_plot=1):
    #字体定义
        font = {'family': 'Times New Roman',
        'color':  'Black',
        'weight': '900',
        'size': 18,
        }
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        #(0,0)处的圆筒
        alpha=np.arange(0,2*np.pi,np.pi/100)
        R=1#半径 
        x=R*np.cos(alpha)
        y=R*np.sin(alpha)
        ax.plot(x,y,'k-',lw=1.5)
        ax.fill(x,y,'navajowhite')
        
        #杆1
        A1 = np.array([[np.cos(q[2,0]),-np.sin(q[2,0])],[np.sin(q[2,0]),np.cos(q[2,0])]])
        S1 = np.array([[-2],[0]])
        S2 = np.array([[2],[0]])
        r1 = (q[0:2]+np.dot(A1,S1)).T
        r2 = (q[0:2]+np.dot(A1,S2)).T
        #print(r2)
        xdata_1 = [r1[0,0], r2[0,0]]
        ydata_1 = [r1[0,1], r2[0,1]]
        line1, = ax.plot(xdata_1, ydata_1, 'b', lw=2.5) #线宽lw=2
        
        #地平面
        line2, =ax.plot([-5,5],[0,0],  'r--', lw=1)
        
        #左侧墙面
        line3, =ax.plot([-2,-2],[-3,3], 'k-', lw=2)
        
        plt.axis('equal')
        
        ''' plt.title('Equilibrium Configuration',fontdict=font) '''
        
        plt.show()
        
        if save_plot==1:

            fig.savefig("plt3.png")
    
               