# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division
try:
    import faiss
except:
    pass
import numpy as np
import torch
import argparse
import os
import time
from lib.metrics import evaluate,sanitize
from lib.net import Normalize,forward_pass
join = os.path.join
import torch.nn as nn
from lib.data import load_dataset

def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)


def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))

if __name__ == "__main__":
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--database", choices=["bigann", "deep1b","mmu10m"])
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"],
                        default="auto")
    parser.add_argument("--gpu", action='store_true', default=False)
    parser.add_argument("--out_prefix", type=str, required=True)
    parser.set_defaults(gpu=False, test=True)

    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    start = time.time()
    print("args.device:{}".format(args.device))
    if os.path.exists(args.ckpt_path):
        train_size=10000
        print("Loading net")
        ckpt = torch.load(args.ckpt_path)
        d = vars(args)
        for k, v in vars(ckpt['args']).items():
            d[k] = v

        (xt, xb, xq, gt) = load_dataset(args.database, args.device, size=args.size_base, test=args.test)
        dim = xb.shape[1]
        dint, dout = args.dint, args.dout

        net = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dint, bias=True),
            nn.BatchNorm1d(dint),
            nn.ReLU(),
            nn.Linear(in_features=dint, out_features=dint, bias=True),
            nn.BatchNorm1d(dint),
            nn.ReLU(),
            nn.Linear(in_features=dint, out_features=dout, bias=True),
            Normalize()
        )
        net.load_state_dict(ckpt['state_dict'])
        net = net.to(args.device)
        net = net.eval()
        xq = forward_pass(net, sanitize(xq), device=args.device)
        print("xq finish")
        xb = forward_pass(net, sanitize(xb), device=args.device)
        xt = forward_pass(net, sanitize(xt), device=args.device)
        fvecs_write(args.out_prefix+".spreadvector.query.d"+str(xq.shape[1])+".n"+str(xq.shape[0])+".fvecs",xq)
        fvecs_write(args.out_prefix+".spreadvector.base.d"+str(xb.shape[1])+".n"+str(xb.shape[0])+".fvecs",xb)
        fvecs_write(args.out_prefix+".spreadvector.learn.d"+str(xt.shape[1])+".n"+str(xt.shape[0])+".fvecs",xt)

    elif args.ckpt_path.startswith("pca-"):
        train_size=100000
        assert args.database is not None
        print("in pca-")
        (xt, xb, xq, gt) = load_dataset(args.database, args.device, size=args.size_base, test=args.test)
        args.dim = int(args.ckpt_path[4:])

        xbtmp = xb[:min(100000,xb.shape[0])] 
        print("in xbtmp shape {}".format(xbtmp.shape))
        mu = np.mean(xbtmp, axis=0, keepdims=True)
        xb -= mu
        xq -= mu

        cov = np.dot(xbtmp.T, xbtmp) / xbtmp.shape[0]
        eigvals, eigvecs = np.linalg.eig(cov)
        o = eigvals.argsort()[::-1]
        PCA = eigvecs[:, o[:args.dim]].astype(np.float32)

        print("calc")
        xb = np.dot(xb, PCA)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        xq = np.dot(xq, PCA)
        xq /= np.linalg.norm(xq, axis=1, keepdims=True)
        xt = np.dot(xt, PCA)
        xt /= np.linalg.norm(xt, axis=1, keepdims=True)
        fvecs_write(args.out_prefix+".pca.query.d"+str(xq.shape[1])+".n"+str(xq.shape[0])+".fvecs",xq)
        fvecs_write(args.out_prefix+".pca.base.d"+str(xb.shape[1])+".n"+str(xb.shape[0])+".fvecs",xb)
        fvecs_write(args.out_prefix+".pca.learn.d"+str(xt.shape[1])+".n"+str(xt.shape[0])+".fvecs",xt)
    else:
        print("Main argument not understood: should be the path to a net checkpoint")
        import sys;sys.exit(1)

