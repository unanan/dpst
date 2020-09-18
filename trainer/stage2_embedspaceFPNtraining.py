import time
import logging
import trainer
from dataset import *


class EmbedSpaceFPNTrainer(trainer.Trainer):
    def __init__(self,**kwargs):
        '''
        Trainer of stage 2 Embed Space FPN Training.

        **The optional arguments are as follow:
        :param :
        '''
        # Set & init the parameter configs
        self.args=trainer.ParseArgs(stage="stage2",**kwargs)

        # Load Dataset
        # self.trainbatches = BinaryDataset(img_folder = self.args.args.train_folder, img_size = self.args.args.dmodel)

        # Init the trainer (called the dataloader as batches in the function)
        super(EmbedSpaceFPNTrainer,self).__init__(self.args,num_classes=2)


    def train(self):
        # Start Training
        self.model.train();iter = 0  # iter id start from 1
        for epoch in range(self.max_epoch):

            for batch in self.trainbatches:
                start = time.time()
                iter += 1
                img_tensor, gt_tensor = batch
                self.optimizer.zero_grad()

                # Forwarding
                preds = self.model(img_tensor)

                # Calculate Loss
                loss = self.criterion(preds, gt_tensor)
                loss.backward()
                self.optimizer.step()
                self.losses.update(loss.item(), self.batch_size)

                if iter % self.show_interval == 0:
                    logging.info(
                        f"Training [{epoch}/{self.max_epoch}][{iter}] Loss:{self.losses.avg} Time:{time.time() - start:.1f}s")

                if iter % self.val_interval == 0:
                    pass
                    # vallosses = AverageMeter()
                    # valaccs = AverageMeter()
                    # valstart = time.time()
                    # # Start Validating
                    # for valbatch in enumerate(valloader):
                    #     val_img_tensor, val_label_tensor = valbatch
                    #     # Forwarding
                    #     preds = model(img_tensor)
                    #
                    #     # Calculate Loss
                    #     loss = criterion(preds, label_tensor)
                    #     vallosses.update(loss.item(), args.val_batch_size)
                    #
                    #     # Calculate accuracy metrics
                    #     acc = None  #TODO
                    #     valaccs.update(acc, args.val_batch_size)
                    # logging.info(f"Validating: Loss:{vallosses.avg} Acc:{valaccs.avg} Time:{time.time() - valstart:.1f}s")
                    #
                    # key_points = model(img_tensor)
                    # key_points = torch.sigmoid(key_points)
                    # binary_kmap = key_points.squeeze().cpu().numpy() > args.threshold
                    # kmap_label = label(binary_kmap, connectivity=1)
                    # props = regionprops(kmap_label)
                    # plist = []
                    # for prop in props:
                    #     plist.append(prop.centroid)
                    #
                    # b_points = reverse_mapping(plist, numAngle=args.num_angle, numRho=args.num_rho, size=(400, 400))
                    # size = (img_tensor.shape[2].item(), img_tensor.shape[3].item())
                    # scale_w = size[1] / 400
                    # scale_h = size[0] / 400
                    # for i in range(len(b_points)):
                    #     y1 = int(np.round(b_points[i][0] * scale_h))
                    #     x1 = int(np.round(b_points[i][1] * scale_w))
                    #     y2 = int(np.round(b_points[i][2] * scale_h))
                    #     x2 = int(np.round(b_points[i][3] * scale_w))
                    #     if x1 == x2:
                    #         angle = -np.pi / 2
                    #     else:
                    #         angle = np.arctan((y1 - y2) / (x1 - x2))
                    #     (x1, y1), (x2, y2) = get_boundary_point(y1, x1, angle, size[0], size[1])
                    #     b_points[i] = (y1, x1, y2, x2)
                    #
                    # # # Show current accuracy
                    # # plx.scatter(x, y, rows= 17, cols = 70)
                    # # plx.show()
