# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:26:06 2024

@author: CUPK-K
"""

import torch
from models.StrucFormer import StrucFormer
import argparse
from utils import TensorCPDec



# parsers
parser = argparse.ArgumentParser(description='PyTorch StrucFormer Training')
parser.add_argument('--lr', default=1e-4, type=float) 
parser.add_argument('--nhead', default=8, type=int) 
parser.add_argument('--depth', default=6, type=int) 
parser.add_argument('--dim_k', default=32, type=int)
parser.add_argument('--dim_ff', default=128, type=int)  
parser.add_argument('--channels', default=128, type=int)  
parser.add_argument('--R', default=5, type=int)  
parser.add_argument('--Alpha', default=1e-2, type=float)  
parser.add_argument('--Beta', default=1e-2, type=float)  
parser.add_argument('--Gamma', default=1e-2, type=float)  
parser.add_argument('--Iters', default=2e4, type=int)  




args = parser.parse_args();


tensordec = TensorCPDec(
                R= args.R,
                Alpha = args.Alpha, 
                Beta  = args.Beta, 
                Gamma = args.Gamma, 
                Iters = args.Iters);



net = StrucFormer(
        input_size=8,
        decode_size=1,
        channels=args.channels,
        dim_k=args.dim_k,
        dim_ff=args.dim_ff,
        nhead=args.nhead,
        depth=args.depth);







