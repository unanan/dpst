import os
import time
import json
import argparse
import logging
import numpy as np
# import plotext.plot as plx

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import model
import loss

class ParseArgs:
    def __init__(self,stage=None,config_file=None,**kwargs):
        '''
        Set the parameters for training.
        :param stage: (str) set the stage number.  Value:['stage1' | 'stage2']
        :param config_file: (str) path to the configuration file in json format. Value:e.g.'/home/user/stage1config.json'
        :param kwargs: some optional arguments
        '''

        if not stage and not config_file:
            raise ValueError(f"Must assign one of the options: 'stage', 'config_file'.")


        parser = argparse.ArgumentParser(description=f'Training Configs')
        # Training
        parser.add_argument('--device', default="0,1", type=str, help='device id(s) for data-parallel during training.')
        parser.add_argument('--batch_size', default=6, type=int, help='batch size for training.')
        parser.add_argument('--num_workers', default=4, type=int, help='number of workers for training.')
        parser.add_argument('--max_epoch', default=2000000, type=int, help='number of epoches for training.')
        parser.add_argument('--base_lr', default=0.0001, type=float, help='learning rate at the beginning.')
        parser.add_argument('--momentum', default=0.9, type=float, help='learning rate at the beginning.')
        parser.add_argument('--weight_decay', default=5e-4, type=float, help='learning rate at the beginning.')
        parser.add_argument('--show_interval', default=100, type=int, help='steps(iters) between two training logging output.')
        # Validating
        parser.add_argument('--val_batch_size', default=6, type=int, help='batch size for validating')
        parser.add_argument('--val_interval', default=200, type=int, help='steps(iters) between two validating phase.')

        if stage:
            if stage=="stage1":
                parser.add_argument('--net', default="embednet", type=str, help='embednet | attn')
                parser.add_argument('--dmodel', default=224, type=int, help='output dimension')
                parser.add_argument('--criterion', default="KLDivSparseLoss", type=str, help='KLDivLoss | KLDivSparseLoss')
                parser.add_argument('--resume', default=None, type=str, help='weights path for resuming the training')

                # Datasets
                parser.add_argument('--train_folder', default=None, type=str, help='binary images for training')

                # Miscs
                parser.add_argument('--save_folder', default="./save_folder", type=str, help='')

            elif stage == "stage2":
                parser.add_argument('--backbone', default="MobileNetV3_Small", type=str, help='resnet18 | resnet50 | resnet101 | resnext50 | vgg16 | MobileNetV3_Small | res2net50')
                parser.add_argument('--criterion', default="", type=str, help='')
                parser.add_argument('--resume', default=None, type=str, help='weights path for resuming the training')

                # Datasets
                parser.add_argument('--train_gtfile', default="test/gt.txt", type=str, help='')
                parser.add_argument('--val_gtfile', default="", type=str, help='')

                # Miscs
                parser.add_argument('--save_folder', default="./save_folder", type=str, help='')
            else:
                raise ValueError(f"Unknown value: {stage}.")
        else:
            # with open(config_file) as f:
                # configs = json.load(f)
            # parser.add_argument_group(configs)
            raise NotImplementedError("Not implemented method: to set parameters by config file.")

        self.args = parser.parse_args()

        # TODO
        # for k, v in kwargs.items():
        #     item = getattr(self.args, k)
        #     item = v




class Trainer:
    def __init__(self,args:ParseArgs,**modelargs):
        self.args = args.args
        # Set device
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.device.strip()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")  # Not suggested

        # Set save folder & logging config
        subfolder = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        if not self.args.save_folder or (not os.path.isdir(self.args.save_folder)):
            print("Warning: Not invalid value of 'save_folder', set as default value: './save_folder'..")
            self.save_folder = "./save_folder"
        else:
            self.save_folder = self.args.save_folder
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
        self.save_folder = os.path.join(self.save_folder, subfolder)
        os.mkdir(self.save_folder)
        # TODO:logging

        # Init train stage
        self.__init_stage(self.args,**modelargs)



    def __init_stage(self,args,**modelargs):
        # Create Dataloader
        if hasattr(self,"trainbatches"):
            self.trainbatches=self.__get_loader(self.trainbatches,shuffle=True)
        if hasattr(self,"valbatches"):
            self.valbatches=self.__get_loader(self.valbatches,shuffle=True)

        # Init Net
        self.model = getattr(model,args.net)(**modelargs)
        if args.resume:
            self.model.load_state_dict(torch.load(args.resume))
        else:
            # self.model.apply(self.__init_weights)
            pass #TODO
        # self.model=self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        # Optimizer #TODO
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.args.base_lr,weight_decay=self.args.weight_decay)
        # self.optimizer = optim.SGD(self.model.parameters(),lr=self.args.base_lr,momentum=self.args.momentum,weight_decay=self.args.weight_decay)

        # Loss
        self.criterion = getattr(loss,args.criterion)()
        self.losses = loss.AverageMeter()  # print on log

        self.show_interval = args.show_interval
        self.max_epoch = args.max_epoch
        self.batch_size = args.batch_size
        self.val_interval = args.val_interval

    def __get_loader(self,dataset,shuffle=True):
        data_loader = DataLoader(dataset=dataset, batch_size=self.args.batch_size, shuffle=shuffle,
                                 num_workers=self.args.num_workers, pin_memory=True)
        return data_loader


    def print_args(self):
        logging.info(self.args)

    def save_pth(self, name = "weights_fin.pth"):
        torch.save(self.model.state_dict(), name)

    def train(self):
        raise NotImplementedError()
