import matplotlib.pyplot as plt
import numpy as np
from matplotlib import  colors


import matplotlib
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FixedLocator



def plot_slice(mesh_u, mesh_v,test_predict,fig_fn,config):

    axis_size = config['axis_size']

    fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
    ax = ax.flatten()
    Ex = test_predict[:, 0].reshape([axis_size,axis_size])
    Ey = test_predict[:, 1].reshape([axis_size,axis_size])
    Ez = test_predict[:, 2].reshape([axis_size,axis_size])

    norm1 = matplotlib.colors.Normalize(vmin=-1,vmax=1)
    ax[0].contourf(mesh_u,mesh_v,Ex,100,norm=norm1,cmap="jet")
    ax[1].contourf(mesh_u,mesh_v,Ey,100,norm=norm1,cmap="jet")
    im = ax[2].contourf(mesh_u,mesh_v,Ez,100,norm=norm1,cmap="jet")

    for i in range(3):
        ax[i].set_aspect(1)

    fig.colorbar(im,ax=[ax[0], ax[1],ax[2]],fraction=0.02,pad=0.1) #原型是plt.Figure
    plt.savefig("%s.png"%fig_fn)
    plt.close

    

def plot_waveguide(waveguide_data,waveguide_label,waveguide_predict):
    

    y = waveguide_data[:, 1:2]
    z = waveguide_data[:, 2:3] 

    fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
    ax = ax.flatten()

    Ex = waveguide_predict[:, 0:1]
    Ey = waveguide_predict[:, 1:2]
    Ez = waveguide_predict[:, 2:3]

    diff = Ez - waveguide_label  

    norm1 = matplotlib.colors.Normalize(vmin=0,vmax=1)

    ax[0].scatter(y,z,c=waveguide_label,s=100,cmap="jet") #
    im1 = ax[1].scatter(y,z,c=Ez,s=100,cmap="jet")
    im2 = ax[2].scatter(y,z,c=diff,s=100,cmap="binary")

    ax[0].set_title('True')
    ax[1].set_title('Predict')
    ax[2].set_title('difference')

    

    for i in range(3):
        ax[i].set_aspect(1)

    fig.tight_layout()

    #设置coloarbar的位置
    l = 0.15
    b = 0.15
    w = 0.4
    h = 0.03
    rec1 = [l,b,w,h] #l,b,w,h对应左 下 宽 高
    cbar_ax1 = fig.add_axes(rec1)

    l = 0.75
    b = 0.15
    w = 0.2
    h = 0.03
    rec2 = [l,b,w,h] #l,b,w,h对应左 下 宽 高
    cbar_ax2 = fig.add_axes(rec2)

    fig.colorbar(im1,ax=[ax[0], ax[1]],orientation='horizontal',ticks=FixedLocator([0,0.5,1,1.5,2]),fraction=0.02,pad=0.1,cax=cbar_ax1) #原型是plt.Figure
    fig.colorbar(im2,ax=[ax[2]],orientation='horizontal',ticks=FixedLocator([-0.06,0,0.1]),fraction=0.02,pad=0.1,cax=cbar_ax2) #原型是plt.Figure
    
    fig.suptitle('Ez',x=0.5, y=0.9)

    plt.savefig("waveguide_Ez")
    plt.close

    fig,ax = plt.subplots(1,1,sharex=True,sharey=True)
    im = ax.scatter(y,z,c=Ex,s=100,norm=norm1,cmap="jet")
    ax.set_aspect(1)
    fig.subplots_adjust(left=0.3)
    fig.subplots_adjust(right=0.7)
    fig.suptitle('Ex',x=0.5, y=0.9)
    ###
    l = 0.35
    b = 0.1
    w = 0.3
    h = 0.03
    rec = [l,b,w,h] #l,b,w,h对应左 下 宽 高
    cbar_ax = fig.add_axes(rec)
    fig.colorbar(im,ax=ax,orientation='horizontal',ticks=FixedLocator([-0.005,0,0.005]),fraction=0.02,pad=0.1,cax=cbar_ax) #原型是plt.Figure
    
    plt.savefig("waveguide_Ex")
    plt.close

    fig,ax = plt.subplots(1,1,sharex=True,sharey=True)
    im = ax.scatter(y,z,c=Ey,s=100,norm=norm1,cmap="jet")
    ax.set_aspect(1)
    fig.subplots_adjust(left=0.3)
    fig.subplots_adjust(right=0.7)
    fig.suptitle('Ey',x=0.5, y=0.9)
    ###
    l = 0.35
    b = 0.1
    w = 0.3
    h = 0.03
    rec = [l,b,w,h] #l,b,w,h对应左 下 宽 高
    cbar_ax = fig.add_axes(rec)
    fig.colorbar(im,ax=ax,orientation='horizontal',ticks=FixedLocator([-0.0005,0,0]),fraction=0.02,pad=0.1,cax=cbar_ax) #原型是plt.Figure
    plt.savefig("waveguide_Ey")
    plt.close



def plot_cross_section(test_data,test_predict,config):
    axis_size = config['axis_size']
    test_data = test_data.reshape(axis_size,axis_size,axis_size,3)    
    test_predict = test_predict.reshape(axis_size,axis_size,axis_size,3)
    
    x = test_data[:,:,:, 0]
    y = test_data[:,:,:, 1]
    z = test_data[:,:,:, 2]
    # print(test_data[select_index,:,:, :])#select_index = int(0) 对应第一个分量全为0

    Ex = test_predict[:,:,:, 0]
    Ey = test_predict[:,:,:, 1]
    Ez = test_predict[:,:,:, 2]

    norm1 = matplotlib.colors.Normalize(vmin=-1,vmax=1)
    
    fig = plt.figure()
    select_index = int(axis_size/2.0)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[select_index,...], y[select_index,...], z[select_index,...],c=Ex[select_index,...],s=100,norm=norm1,cmap='jet')
    ax.scatter(x[:,select_index,:], y[:,select_index,:], z[:,select_index,:],c=Ex[:,select_index,:],s=100,norm=norm1,cmap='jet')
    im = ax.scatter(x[:,:,select_index], y[:,:,select_index], z[:,:,select_index],c=Ex[:,:,select_index],s=100,norm=norm1,cmap='jet')  
    
    # ax.set_aspect('equal') #3D不支持
    ax.set_box_aspect([2,2,2]) 
    fig.colorbar(im,ax=ax,fraction=0.02,pad=0.1) #原型是plt.Figure

    plt.savefig("Ex")
    plt.close


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[select_index,...], y[select_index,...], z[select_index,...],c=Ey[select_index,...],s=100,norm=norm1,cmap='jet')
    ax.scatter(x[:,select_index,:], y[:,select_index,:], z[:,select_index,:],c=Ey[:,select_index,:],s=100,norm=norm1,cmap='jet')
    im = ax.scatter(x[:,:,select_index], y[:,:,select_index], z[:,:,select_index],c=Ey[:,:,select_index],s=100,norm=norm1,cmap='jet')  
    
    # ax.set_aspect('equal') #3D不支持
    ax.set_box_aspect([2,2,2]) 
    fig.colorbar(im,ax=ax,fraction=0.02,pad=0.1) #原型是plt.Figure

    plt.savefig("Ey")
    plt.close

    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[select_index,...], y[select_index,...], z[select_index,...],c=Ez[select_index,...],s=100,norm=norm1,cmap='jet')
    ax.scatter(x[:,select_index,:], y[:,select_index,:], z[:,select_index,:],c=Ez[:,select_index,:],s=100,norm=norm1,cmap='jet')
    im = ax.scatter(x[:,:,select_index], y[:,:,select_index], z[:,:,select_index],c=Ez[:,:,select_index],s=100,norm=norm1,cmap='jet')  
    
    # ax.set_aspect('equal') #3D不支持
    ax.set_box_aspect([2,2,2]) 
    fig.colorbar(im,ax=ax,fraction=0.02,pad=0.1) #原型是plt.Figure

    plt.savefig("Ez")
    plt.close


    
    # # index = np.where(test_data[:,0]==0)[0]
    # # print(index.shape)
    # # x_slice0_data = test_data[index]
    # # x_slice0_predict = test_predict[index]
    # # print(x_slice0_data.shape)
    # # print(x_slice0_predict.shape)   

    # norm1 = matplotlib.colors.Normalize(vmin=0,vmax=1)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(x_slice0_data[...,0], x_slice0_data[...,1], x_slice0_data[...,2],c=x_slice0_predict[...,0],s=100,norm=norm1)
    # print(np.min(np.abs(x_slice0_predict)))
    # plt.savefig("test")
    # plt.close



    fig,ax = plt.subplots(3,3,sharex=True,sharey=True)
    ax = ax.flatten()

    ax[0].contourf(y[select_index,...], z[select_index,...],Ex[select_index,...],100,norm=norm1,cmap="jet")
    ax[1].contourf(x[:,select_index,:], z[:,select_index,:],Ex[:,select_index,:],100,norm=norm1,cmap='jet')
    ax[2].contourf(x[:,:,select_index], y[:,:,select_index],Ex[:,:,select_index],100,norm=norm1,cmap='jet')
    
    # print(x[select_index,...]) #第一个index固定为select_index 这个对应的是x不变 第二个固定是y不变 
    ax[0].set_title(r'$E_x,x=$%s'%(x[select_index,0,0]))
    ax[1].set_title(r'$E_x,y=$%s'%(y[0,select_index,0]))
    ax[2].set_title(r'$E_x,z=$%s'%(z[0,0,select_index]))

    

    ax[3].contourf(y[select_index,...], z[select_index,...],Ey[select_index,...],100,norm=norm1,cmap="jet")
    ax[4].contourf(x[:,select_index,:], z[:,select_index,:],Ey[:,select_index,:],100,norm=norm1,cmap='jet')
    ax[5].contourf(x[:,:,select_index], y[:,:,select_index],Ey[:,:,select_index],100,norm=norm1,cmap='jet')
    

    ax[3].set_title(r'$E_y,x=$%s'%(x[select_index,0,0]))
    ax[4].set_title(r'$E_y,y=$%s'%(y[0,select_index,0]))
    ax[5].set_title(r'$E_y,z=$%s'%(z[0,0,select_index]))

    ax[6].contourf(y[select_index,...], z[select_index,...],Ez[select_index,...],100,norm=norm1,cmap="jet")
    ax[7].contourf(x[:,select_index,:], z[:,select_index,:],Ez[:,select_index,:],100,norm=norm1,cmap='jet')
    im = ax[8].contourf(x[:,:,select_index], y[:,:,select_index],Ez[:,:,select_index],100,norm=norm1,cmap='jet')
    

    ax[6].set_title(r'$E_z,x=$%s'%(x[select_index,0,0]))
    ax[7].set_title(r'$E_z,y=$%s'%(y[0,select_index,0]))
    ax[8].set_title(r'$E_z,z=$%s'%(z[0,0,select_index]))

    

    for i in range(9):
        ax[i].set_aspect(1)
    
    # fig.tight_layout()
    fig.subplots_adjust(left=0.125,
                        bottom=0.1, 
                        right=0.8, 
                        top=0.9, 
                        wspace=0.2, 
                        hspace=0.35
                    )
    
    ###
    l = 0.85
    b = 0.2
    w = 0.03
    h = 0.6
    rec = [l,b,w,h] #l,b,w,h对应左 下 宽 高
    cbar_ax = fig.add_axes(rec)
    fig.colorbar(im,ax=ax,fraction=0.02,pad=0.1,cax=cbar_ax) #原型是plt.Figure
    

    plt.savefig("E_cross_section")
    plt.close
