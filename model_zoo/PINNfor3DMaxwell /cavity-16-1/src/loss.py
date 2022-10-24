import numpy as np

import mindspore as ms
from mindspore import Tensor

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import  ms_function


from mindelec.operators import SecondOrderGrad as Hessian
from mindelec.operators import Grad
import mindspore.numpy as ms_np


class CustomWithLossCell(nn.Cell):
    def __init__(self, net, Maxwell_3d_config):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self.net = net
        self.wave_number = Maxwell_3d_config["wave_number"]
        self.grad_ux_xx = Hessian(net, input_idx1=0, input_idx2=0, output_idx=0)
        self.grad_ux_yy = Hessian(net, input_idx1=1, input_idx2=1, output_idx=0)
        self.grad_ux_zz = Hessian(net, input_idx1=2, input_idx2=2, output_idx=0)
        self.grad_uy_xx = Hessian(net, input_idx1=0, input_idx2=0, output_idx=1)
        self.grad_uy_yy = Hessian(net, input_idx1=1, input_idx2=1, output_idx=1)
        self.grad_uy_zz = Hessian(net, input_idx1=2, input_idx2=2, output_idx=1)
        self.grad_uz_xx = Hessian(net, input_idx1=0, input_idx2=0, output_idx=2)
        self.grad_uz_yy = Hessian(net, input_idx1=1, input_idx2=1, output_idx=2)
        self.grad_uz_zz = Hessian(net, input_idx1=2, input_idx2=2, output_idx=2)
        self.reshape = ops.Reshape()
        self.print = ops.Print()
        self.concat = ops.Concat(axis=1)
        self.l2_loss = nn.MSELoss()
        self.zeros_like = ops.ZerosLike()
        self.reduce_mean = ops.ReduceMean()


        self.sin = ops.Sin()
        self.k = Tensor(Maxwell_3d_config["eigenmode"], ms.float32) #eigenmode
        self.pi = Tensor(np.pi, ms.float32)
        self.coord_min = Tensor(Maxwell_3d_config["coord_min"],ms.float32)
        self.coord_max = Tensor(Maxwell_3d_config["coord_max"],ms.float32)
        axis_len = self.coord_max - self.coord_min
        self.width  =  axis_len[0]
        self.height =  axis_len[2]


        self.grad = Grad(net)
        self.zeros =  ops.Zeros()

        self.weight_in_domain = Tensor(1.0 /Maxwell_3d_config["wave_number"]**2,ms.float32)
        self.weight_bc = Tensor(Maxwell_3d_config["weight_bc"],ms.float32)

        # self.weight_waveguide = Tensor(Maxwell_3d_config["weight_waveguide"],ms.float32)
        # self.weight_ABC = Tensor(Maxwell_3d_config["weight_ABC"],ms.float32)
        # self.weight_PEC = Tensor(Maxwell_3d_config["weight_PEC"],ms.float32)

        self.print = ops.Print()

        self.mul = ops.Mul()
        
        

    def construct(self, in_domain_data,bc_data):

        ####
        u_in_domain =  self.net(in_domain_data)
        in_domain_loss = self.governing_equation(u_in_domain, in_domain_data)

        u_bc = self.net(bc_data)

        bc_loss = self.bc_loss_func(u_bc, bc_data)


        total_loss = self.weight_in_domain * in_domain_loss +  self.weight_bc * bc_loss
        
        return total_loss


    @ms_function
    def governing_equation(self, u, input):
        """governing equation"""

        # self.print('#####')
        # self.print(input.shape)
        
        ux_xx = self.grad_ux_xx(input)
        ux_yy = self.grad_ux_yy(input)
        ux_zz = self.grad_ux_zz(input)

        uy_xx = self.grad_uy_xx(input)
        uy_yy = self.grad_uy_yy(input)
        uy_zz = self.grad_uy_zz(input)

        uz_xx = self.grad_uz_xx(input)
        uz_yy = self.grad_uz_yy(input)
        uz_zz = self.grad_uz_zz(input)

        #暂时不考虑slab

        u_xx = self.concat((ux_xx,uy_xx,uz_xx))
        u_yy = self.concat((ux_yy,uy_yy,uz_yy))
        u_zz = self.concat((ux_zz,uy_zz,uz_zz))
        # self.print(u_xx.shape) #这个loss function定义有问题 应该是三维的
        equa = u_xx + u_yy + u_zz + self.wave_number**2 * u
        # self.print(equa.shape)

        in_domain_loss = self.reduce_mean(self.l2_loss(equa, self.zeros_like(equa)))

        #无源场 自然应该是有div等于0？是否要加

        ux_x = self.grad(input,0,0,u)

        uy_y = self.grad(input,1,1,u)

        uz_z = self.grad(input,2,2,u)

        div = ux_x + uy_y + uz_z

        nosource_loss = self.reduce_mean(self.l2_loss(div, self.zeros_like(div)))
        

        return in_domain_loss+nosource_loss

    def bc_loss_func(self, u, input):

        y = input[:, 1:2]
        z = input[:, 2:]   
        waveguide_port = self.sin(self.k * self.pi * y / self.width) * self.sin(self.k * self.pi * z / self.height)
        ux = u[:,0:1]
        uy = u[:,1:2]
        uz = u[:,2:]
        pre_Left_constraint =  self.concat((ux, uy, 10.0*(uz-waveguide_port)))

        select_index = ms_np.zeros(shape=(input.shape[0],17),dtype=ms.float32)

        select_index[:,0] = ms_np.where(ms_np.equal(input[:,0],self.coord_min[0]),1.0,0.0)
        select_index[:,1] = ms_np.where(ms_np.equal(input[:,0],self.coord_min[0]),1.0,0.0)
        select_index[:,2] = ms_np.where(ms_np.isclose(input[:,0],self.coord_min[0]),1.0,0.0)
        
        # self.print(self.reduce_mean(waveguide_port**2))
        # is_bc_Left =  ms_np.where(ms_np.equal(input[:,0:1],self.coord_min[0]),1.0,0.0) # 用来判断是否数据在L边界上 如果在是1，否则是0
        # Left_constraint = pre_Left_constraint*is_bc_Left.broadcast_to(pre_Left_constraint.shape)        
        # waveguide_loss = self.reduce_mean(self.l2_loss(Left_constraint, self.zeros_like(Left_constraint)))

        ux_z = self.grad(input,2,0,u)
        uz_x = self.grad(input,0,2,u)
        uy_x = self.grad(input,0,1,u)
        ux_y = self.grad(input,1,0,u)

        cross_product_Grad_E_y = ux_z  - uz_x#grad_z(ux) - grad_x(uz)
        cross_product_Grad_E_z = uy_x  - ux_y
        
        pre_Right_constraint = self.concat((cross_product_Grad_E_y, cross_product_Grad_E_z))     
        # pre_Right_constraint = self.ABC_constraint(u, input,normal_vector=Tensor([1.0,0.0,0.0],ms.float32))
             
        # is_bc_Right =  ms_np.where(ms_np.equal(input[:,0:1],self.coord_max[0]),1.0,0.0) # 用来判断是否数据在R边界上 如果在是1，否则是0
        # Right_constraint = pre_Right_constraint*is_bc_Right.broadcast_to(pre_Right_constraint.shape)        
        # ABC_loss = self.reduce_mean(self.l2_loss(Right_constraint, self.zeros_like(Right_constraint)))
        select_index[:,3] = ms_np.where(ms_np.isclose(input[:,0],self.coord_max[0]),1.0,0.0)
        select_index[:,4] = ms_np.where(ms_np.isclose(input[:,0],self.coord_max[0]),1.0,0.0)
    
       
        uy_y = self.grad(input,1,1,u)
        uz_z = self.grad(input,2,2,u)

        pre_Front_constraint = self.concat((uz,ux,uy_y))
        # pre_Front_constraint = self.PEC_constraint(u, input,normal_vector=Tensor([0.0,-1.0,0.0],ms.float32))
        # is_bc_Front =  ms_np.where(ms_np.equal(input[:,1:2],self.coord_min[1]),1.0,0.0) # 用来判断是否数据在F边界上 如果在是1，否则是0
        # Front_constraint = pre_Front_constraint*is_bc_Front.broadcast_to(pre_Front_constraint.shape)        
        # Front_loss = self.reduce_mean(self.l2_loss(Front_constraint, self.zeros_like(Front_constraint)))
        select_index[:,5] = ms_np.where(ms_np.isclose(input[:,1],self.coord_min[1]),1.0,0.0)
        select_index[:,6] = ms_np.where(ms_np.isclose(input[:,1],self.coord_min[1]),1.0,0.0)
        select_index[:,7] = ms_np.where(ms_np.isclose(input[:,1],self.coord_min[1]),1.0,0.0)

        # pre_Back_constraint = self.PEC_constraint(u, input,normal_vector=Tensor([0.0,1.0,0.0],ms.float32))
       
        pre_Back_constraint = self.concat((uz,ux,uy_y))        
        # is_bc_Back =  ms_np.where(ms_np.equal(input[:,1:2],self.coord_max[1]),1.0,0.0) # 用来判断是否数据在Back边界上 如果在是1，否则是0
        # Back_constraint = pre_Back_constraint*is_bc_Back.broadcast_to(pre_Back_constraint.shape)        
        # Back_loss = self.reduce_mean(self.l2_loss(Back_constraint, self.zeros_like(Back_constraint)))
        select_index[:,8] = ms_np.where(ms_np.isclose(input[:,1],self.coord_max[1]),1.0,0.0)
        select_index[:,9] = ms_np.where(ms_np.isclose(input[:,1],self.coord_max[1]),1.0,0.0)
        select_index[:,10] = ms_np.where(ms_np.isclose(input[:,1],self.coord_max[1]),1.0,0.0)
     
        pre_Bottom_constraint = self.concat((uy,ux,uz_z))
        # pre_Bottom_constraint = self.PEC_constraint(u, input,normal_vector=Tensor([0.0,0.0,-1.0],ms.float32))        
        # is_bc_Bottom =  ms_np.where(ms_np.equal(input[:,2:3],self.coord_min[2]),1.0,0.0) # 用来判断是否数据在Bottom边界上 如果在是1，否则是0
        # Bottom_constraint = pre_Bottom_constraint*is_bc_Bottom.broadcast_to(pre_Bottom_constraint.shape)        
        # Bottom_loss = self.reduce_mean(self.l2_loss(Bottom_constraint, self.zeros_like(Bottom_constraint)))
        select_index[:,11] = ms_np.where(ms_np.isclose(input[:,2],self.coord_min[2]),1.0,0.0)
        select_index[:,12] = ms_np.where(ms_np.isclose(input[:,2],self.coord_min[2]),1.0,0.0)
        select_index[:,13] = ms_np.where(ms_np.isclose(input[:,2],self.coord_min[2]),1.0,0.0)
     
        pre_Top_constraint = self.concat((uy,ux,uz_z))
        # pre_Top_constraint = self.PEC_constraint(u, input,normal_vector=Tensor([0.0, 0.0, 1.0],ms.float32))     
        # is_bc_Top =  ms_np.where(ms_np.equal(input[:,2:3],self.coord_max[2]),1.0,0.0) # 用来判断是否数据在T边界上 如果在是1，否则是0
        # Top_constraint = pre_Top_constraint*is_bc_Top.broadcast_to(pre_Top_constraint.shape)        
        # Top_loss = self.reduce_mean(self.l2_loss(Top_constraint, self.zeros_like(Top_constraint)))
        select_index[:,14] = ms_np.where(ms_np.isclose(input[:,2],self.coord_max[2]),1.0,0.0)
        select_index[:,15] = ms_np.where(ms_np.isclose(input[:,2],self.coord_max[2]),1.0,0.0)
        select_index[:,16] = ms_np.where(ms_np.isclose(input[:,2],self.coord_max[2]),1.0,0.0)

        pre_constraint = self.concat((pre_Left_constraint,
                                      pre_Right_constraint,
                                      pre_Front_constraint,
                                      pre_Back_constraint,
                                      pre_Bottom_constraint,
                                      pre_Top_constraint,
                                    ))
        # bc_loss = self.weight_waveguide * waveguide_loss \
        #             + self.weight_ABC * ABC_loss \
        #             +  self.weight_PEC * (Front_loss + Back_loss + Bottom_loss + Top_loss)
        bc_constraints = self.mul(pre_constraint,select_index)

        bc_loss = self.reduce_mean(self.l2_loss(bc_constraints, self.zeros_like(bc_constraints)))


        return   bc_loss
    
    # @ms_function #可以删掉 本身是cell
    # def waveguide_constraint(self, u, input):

    #     y = input[:, 1:2]
    #     z = input[:, 2:]   
    #     waveguide_port = self.sin(self.k * self.pi * y / self.width) * self.sin(self.k * self.pi * z / self.height)

    #     ux = u[:,0:1]
    #     uy = u[:,1:2]
    #     uz = u[:,2:]

    #     wave_guide = self.concat((ux, uy, uz-waveguide_port))
    #     # waveguide_loss = self.reduce_mean(self.l2_loss(wave_guide, self.zeros_like(wave_guide)))
        
    #     return wave_guide

    
    # @ms_function
    # def ABC_constraint(self, u, input, normal_vector):
 
    #     cross_product_Grad_E_x = self.grad(input,1,2,u)  - self.grad(input,2,1,u)
    #     cross_product_Grad_E_y = self.grad(input,2,0,u)  - self.grad(input,0,2,u)#grad_z(ux) - grad_x(uz)
    #     cross_product_Grad_E_z = self.grad(input,0,1,u)  - self.grad(input,1,0,u)
        
    #     n1 = normal_vector[0]
    #     n2 = normal_vector[1]
    #     n3 = normal_vector[2]

    #     cross_product_n_Grad_E_x = n2*cross_product_Grad_E_z - cross_product_Grad_E_y*n3
    #     cross_product_n_Grad_E_y = n3*cross_product_Grad_E_x - cross_product_Grad_E_z*n1
    #     cross_product_n_Grad_E_z = n1*cross_product_Grad_E_y - cross_product_Grad_E_x*n2
        
    #     cross_product_n_Grad_E = self.concat((cross_product_n_Grad_E_x, cross_product_n_Grad_E_y, cross_product_n_Grad_E_z))
        
    #     # ABC_loss = self.reduce_mean(self.l2_loss(cross_product_n_Grad_E, self.zeros_like(cross_product_n_Grad_E)))    
        
    #     return   cross_product_n_Grad_E
    

    # @ms_function
    # def PEC_constraint(self, u, input, normal_vector):


    #     n1 = normal_vector[0]
    #     n2 = normal_vector[1]
    #     n3 = normal_vector[2]

    #     ux = u[:,0:1]
    #     uy = u[:,1:2]
    #     uz = u[:,2:]

    #     cross_product_n_E_x = n2*uz - uy*n3
    #     cross_product_n_E_y = n3*ux - uz*n1
    #     cross_product_n_E_z = n1*uy - ux*n2

    #     ux_x = self.grad(input,0,0,u)

    #     uy_y = self.grad(input,1,1,u)

    #     uz_z = self.grad(input,1,1,u)

    #     #n div grad_E = 0

    #     div = ux_x*n1 + uy_y * n2 + uz_z*n3



    #     PEC_con = self.concat((cross_product_n_E_x,cross_product_n_E_y,cross_product_n_E_z,div))

        
    #     # PEC_loss = self.reduce_mean(self.l2_loss(cross_product_n_E, self.zeros_like(cross_product_n_E)))    

    #     return   PEC_con



    