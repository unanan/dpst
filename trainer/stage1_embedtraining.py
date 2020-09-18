import time
import logging
import trainer
from dataset import *
import torch.nn as nn

class EmbedTrainer(trainer.Trainer):
    def __init__(self,**kwargs):
        '''
        Trainer of stage 1 Embed Training.

        **The optional arguments are as follow:
        :param :
        '''
        # Set & init the parameter configs
        self.args=trainer.ParseArgs(stage="stage1",**kwargs)

        # Load Dataset
        self.trainbatches = BinaryDataset(img_size = self.args.args.dmodel)

        # Init the trainer (TODO: call the dataloader as batches in the function)
        super(EmbedTrainer,self).__init__(self.args,d_model=self.args.args.dmodel)



    def train(self):
        '''
        Stage 1 training: No validating phase
        This stage train for init weights for stage 2.
        '''
        # Start Training
        self.model.train();iter = 0  # iter id start from 1
        for epoch in range(self.max_epoch):

            for batch in self.trainbatches:
                start = time.time()
                iter += 1
                img_tensors,hotnum = batch
                imgtensor,posimgtensor,negimgtensor = img_tensors
                self.optimizer.zero_grad()

                # Forwarding
                imgmasks = self.model(imgtensor)
                posimgmasks = self.model(posimgtensor)
                negimgmasks = self.model(negimgtensor)

                # Calculate Loss
                loss = self.criterion(imgmasks, posimgmasks, negimgmasks, hotnum)
                # print("DEBUG over")
                loss.backward()
                self.optimizer.step()
                self.losses.update(loss.item(), self.batch_size)

                if iter % self.show_interval == 0:
                    print(
                        f"Training [{epoch}/{self.max_epoch}][{iter}] Loss:{self.losses.avg} Time:{time.time() - start:.1f}s")
                    self.save_pth(f"test.pth")
        self.save_pth(f"stage1_{self.losses.avg:.2f}.pth")