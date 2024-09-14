# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 17:39:37 2024

@author: CUPK-K
"""



import torch
import torch.nn as nn



device = 'cuda' if torch.cuda.is_available() else 'cpu'




class TensorCPDec:
    def __init__(self, R = 5, Alpha = 0.01, Beta = 0.01, Gamma = 0.01, Iters = 2e4):
  

        self.R = R;
     
        
        self.MSELoss = nn.MSELoss();
        self.Alpha = Alpha;
        self.Beta  = Beta;
        self.Gamma = Gamma;
        
        self.loss = None;
        self.LossArray = [];
        self.Iters = int(Iters);
        
         

        if device == 'cuda':
            self.A.cuda();self.B.cuda();self.C.cuda();
            self.MSELoss.cuda();

        
        
    def infer(self):
        I,J,K = self.input_tensor.shape;
        tensor_inf = torch.zeros([I,J,K]);
        
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    tensor_inf[i,j,k] = torch.sum( torch.mul( torch.mul( self.A[i,:] , self.B[j,:]) , self.C[k,:]) );

        return tensor_inf.detach();


    def train(self,input_tensor,mask_tensor):
        self.input_tensor = input_tensor;
        self.A = nn.Parameter(torch.randn(input_tensor.shape[0],self.R));
        self.B = nn.Parameter(torch.randn(input_tensor.shape[1],self.R));
        self.C = nn.Parameter(torch.randn(input_tensor.shape[2],self.R));  
        self.nonzero_indices = torch.nonzero(mask_tensor).numpy();
        self.optimizer = torch.optim.Adam([ self.A,self.B ,self.C ], lr=1e-4) ;
        

        for _ in range(self.Iters):
            
            iter_loss = torch.zeros([1]);
            for ind in self.nonzero_indices:
                i,j,k = ind[0],ind[1],ind[2];
                
                tensor_inf = torch.sum( torch.mul( torch.mul( self.A[i,:] , self.B[j,:]) , self.C[k,:]) );
                ls = torch.norm( (self.input_tensor[i,j,k]- tensor_inf), p="fro");
                iter_loss += ls;
                
            iter_loss = iter_loss + self.Alpha * torch.norm(self.A, p="fro")  \
                                  + self.Beta  * torch.norm(self.B, p="fro")  \
                                  + self.Gamma * torch.norm(self.C, p="fro");
            
            self.optimizer.zero_grad();          
            iter_loss.backward(); 
            self.optimizer.step();
            
            self.LossArray.append(iter_loss.detach().numpy()[0]);
            print('Iters:%d, Loss: %.4f' % (len(self.LossArray),iter_loss.detach().numpy()) );            
            
           


























