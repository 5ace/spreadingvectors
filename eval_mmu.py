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
from lib.metrics import evaluate
from lib.net import Normalize
join = os.path.join
import torch.nn as nn
from lib.data import load_dataset


if __name__ == "__main__":
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--database", choices=["bigann", "deep1b", "mmu"])
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"],
                        default="auto")
    parser.add_argument("--gpu", action='store_true', default=False)
    parser.add_argument("--quantizer", required=True)
    parser.add_argument("--size-base", type=int, default=int(1e9))
    #parser.add_argument("--size-base", type=int, required=True)
    parser.add_argument("--val", action='store_false', dest='test')
    parser.set_defaults(gpu=False, test=True)

    args = parser.parse_args()
    print("args.size-base=%d", args.size_base)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    start = time.time()
    if os.path.exists(args.ckpt_path):
        print("Loading net")
        ckpt = torch.load(args.ckpt_path,map_location='cpu')
        d = vars(args)
        for k, v in vars(ckpt['args']).items():
            d[k] = v
        (xt, xb, xq, gt) = load_dataset(args.database, args.device, size=args.size_base, test=args.test)
        dim = xb.shape[1]
        dint, dout = args.dint, args.dout
        args.size_base = 1

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

        f = open('model.txt', 'w')
        np.savetxt(f, net.state_dict()["0.weight"])
        np.savetxt(f, net.state_dict()["0.bias"])
        np.savetxt(f, net.state_dict()["1.weight"])
        np.savetxt(f, net.state_dict()["1.bias"])
        np.savetxt(f, net.state_dict()["1.running_mean"])
        np.savetxt(f, net.state_dict()["1.running_var"])
        np.savetxt(f, net.state_dict()["3.weight"])
        np.savetxt(f, net.state_dict()["3.bias"])
        np.savetxt(f, net.state_dict()["4.weight"])
        np.savetxt(f, net.state_dict()["4.bias"])
        np.savetxt(f, net.state_dict()["4.running_mean"])
        np.savetxt(f, net.state_dict()["4.running_var"])
        np.savetxt(f, net.state_dict()["6.weight"])
        np.savetxt(f, net.state_dict()["6.bias"])
        f.close()

        net = net.to(args.device)
        net = net.eval()

    elif args.ckpt_path.startswith("pca-"):
        assert args.database is not None
        (xt, xb, xq, gt) = load_dataset(args.database, args.device, size=args.size_base, test=args.test)
        args.dim = int(args.ckpt_path[4:])

        mu = np.mean(xb, axis=0, keepdims=True)
        xb -= mu
        xq -= mu
        #xt -= mu

        cov = np.dot(xb.T, xb) / xb.shape[0]
        eigvals, eigvecs = np.linalg.eig(cov)
        o = eigvals.argsort()[::-1]
        PCA = eigvecs[:, o[:args.dim]].astype(np.float32)

        xb = np.dot(xb, PCA)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        xq = np.dot(xq, PCA)
        xq /= np.linalg.norm(xq, axis=1, keepdims=True)
        #xt = np.dot(xt, PCA)
        #xt /= np.linalg.norm(xt, axis=1, keepdims=True)
        net = nn.Sequential()
    elif args.ckpt_path.startswith("mmupca-"):
        assert args.database is not None
        (xt, xb, xq, gt) = load_dataset(args.database, args.device, size=args.size_base, test=args.test)
        args.dim = int(args.ckpt_path[7:])

        net = nn.Sequential()
    else:
        print("Main argument not understood: should be the path to a net checkpoint")
        import sys;sys.exit(1)

    evaluate(net, xq, xb, gt, [args.quantizer], '%s,rank=%d' % (args.quantizer, 10), device=args.device, trainset=xb[:200000])
