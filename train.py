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

    obj_batch = [
                ['JDTuni100'],
                ['tree_extended'],
                ['tree'],
                ['capsule'],
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
        obj_list = ['JDTuni100',
                    'tree_extended',
                    'tree',
                    'capsule',
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
    os.environ['MASTER_PORT'] = '12355'
    train.train(args)