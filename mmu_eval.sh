python -u eval.py --quantizer none --ckpt-path /home/zhangcunyi/code/spreadingvectors/1024_cata64_none_iter160/checkpoint.pth.best   --database mmu10m &>log.eval.iter160.none
python -u eval.py --quantizer opq16 --ckpt-path /home/zhangcunyi/code/spreadingvectors/1024_cata64_none_iter160/checkpoint.pth.best   --database mmu10m &>log.eval.iter160.opq16
python -u eval.py --quantizer opq16 --ckpt-path pca-128   --database mmu10m &>log.eval.pca.opq16
