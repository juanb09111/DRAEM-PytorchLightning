# %%
import os
import os.path
import sys
import torch.multiprocessing as mp
from argparse import ArgumentParser
from train_scripts import train
        

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True, help="Name of the experiment")
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--config', type=str, required=True, help="Config file from configs/")
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--fast_dev', action='store_true')

    args = parser.parse_args()

    obj_batch = [['capsule'],
                 ['bottle'],
                 ['carpet'],
                 ['leather'],
                 ['pill'],
                 ['transistor'],
                 ['tile'],
                 ['cable'],
                 ['zipper'],
                 ['toothbrush'],
                 ['metal_nut'],
                 ['hazelnut'],
                 ['screw'],
                 ['grid'],
                 ['wood']
                 ]

    if int(args.obj_id) == -1:
        obj_list = ['capsule',
                     'bottle',
                     'carpet',
                     'leather',
                     'pill',
                     'transistor',
                     'tile',
                     'cable',
                     'zipper',
                     'toothbrush',
                     'metal_nut',
                     'hazelnut',
                     'screw',
                     'grid',
                     'wood'
                     ]
        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]
    
    args.obj_names = picked_classes
    # add the ip address to the environment variable so it can be easily avialbale
    # os.environ['MASTER_ADDR'] = args.ip_adress
    # print("ip_adress is", args.ip_adress)
    os.environ['MASTER_PORT'] = '12355'
    train.train(args)