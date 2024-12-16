import argparse
import os, time
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from model.sync_batchnorm.replicate import patch_replication_callback
from model.coanet import *
from utils.loss import SegmentationLosses, dice_bce_loss,soft_dice_cldice
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = CoANet(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_2x_lr_params(), 'lr': args.lr * 2}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = dice_bce_loss()
        self.centralLoss=soft_dice_cldice()
        self.criterion_con = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        self.evaluator = Evaluator(2)


        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            print(checkpoint['best_pred'])
            args.start_epoch = checkpoint['epoch']

            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])

            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss1 = 0.0
        train_loss2 = 0.0
        train_loss3 = 0.0
        train_losscl=0.0
        train_loss = 0.0
        self.model.train()
        self.evaluator.reset()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            image, target, con0, con1, con2, con3, con4, con_d1_0, con_d1_1, con_d1_2, con_d1_3, con_d1_4 = \
                sample['image'], sample['label'], sample['connect0'], sample['connect1'], sample['connect2'], sample['connect3'], sample['connect4'],\
                sample['connect_d1_0'], sample['connect_d1_1'], sample['connect_d1_2'], sample['connect_d1_3'], sample['connect_d1_4']
            connect_label = torch.cat((con0[:,0,:,:].reshape(-1,1,512,512), con1[:,0,:,:].reshape(-1,1,512,512), con2[:,0,:,:].reshape(-1,1,512,512), con3[:,0,:,:].reshape(-1,1,512,512), con4[:,0,:,:].reshape(-1,1,512,512)), 1)
            connect_d1_label = torch.cat((con_d1_0[:,0,:,:].reshape(-1,1,512,512), con_d1_1[:,0,:,:].reshape(-1,1,512,512), con_d1_2[:,0,:,:].reshape(-1,1,512,512), con_d1_3[:,0,:,:].reshape(-1,1,512,512), con_d1_4[:,0,:,:].reshape(-1,1,512,512)), 1)
            if self.args.cuda:
                image, target, connect_label, connect_d1_label = image.cuda(), target.cuda(), connect_label.cuda(), connect_d1_label.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output, out_connect, out_connect_d1 = self.model(image)
            target = torch.unsqueeze(target, 1)

            loss1 = self.criterion(output, target)
            #中心线loss
            if epoch<800:
                loss_cl=0
            else:
                loss_cl=self.centralLoss(output.clone(),target)
                train_losscl+=loss_cl.item()
            loss2 = self.criterion_con(out_connect, connect_label)
            loss3 = self.criterion_con(out_connect_d1, connect_d1_label)
            lad = 0.2
            loss = loss1 + lad*(0.6*loss2 + 0.4*loss3)+loss_cl
            loss.backward()
            self.optimizer.step()
            train_loss1 += loss1.item()
            train_loss2 += lad * 0.6 * loss2.item()
            train_loss3 += lad * 0.4 * loss3.item()
            #train_losscl+=loss_cl.item()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f, loss1: %.6f, loss2: %.3f, loss3: %.3f, losscl: %.3f' %
                                 (train_loss / (i + 1), train_loss1 / (i + 1), train_loss2 / (i + 1), train_loss3 / (i + 1),train_losscl/(i+1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            pred = output.data.cpu().numpy()
            target_n = target.cpu().numpy()
            # Add batch sample into evaluator
            pred[pred > 0.1]=1
            pred[pred < 0.1] = 0
            self.evaluator.add_batch(target_n, pred)

        # Acc = self.evaluator.Pixel_Accuracy()
        # Acc_class = self.evaluator.Pixel_Accuracy_Class()
        # mIoU = self.evaluator.Mean_Intersection_over_Union()
        # IoU = self.evaluator.Intersection_over_Union()
        # Precision = self.evaluator.Pixel_Precision()
        # Recall = self.evaluator.Pixel_Recall()
        # F1 = self.evaluator.Pixel_F1()
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        self.writer.add_scalar('train/loss1_epoch', train_loss1/num_img_tr, epoch)
        self.writer.add_scalar('train/loss2_epoch', train_loss2, epoch)
        self.writer.add_scalar('train/train_losscl', train_losscl, epoch)
        self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], epoch)
        # self.writer.add_scalar('train/mIoU', mIoU, epoch)
        # self.writer.add_scalar('train/Acc', Acc, epoch)
        # self.writer.add_scalar('train/Acc_class', Acc_class, epoch)
        # self.writer.add_scalar('train/IoU', IoU, epoch)
        # self.writer.add_scalar('train/Precision', Precision, epoch)
        # self.writer.add_scalar('train/Recall', Recall, epoch)
        # self.writer.add_scalar('train/F1', F1, epoch)
        print('Train:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        # print("Acc:{}, Acc_class:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{}"
        #       .format(Acc, Acc_class, mIoU, IoU, Precision, Recall, F1))
        print('Loss: %.3f, Loss1: %.6f, Loss2: %.3f, Loss3: %.3f' % (train_loss, train_loss1, train_loss2, train_loss2))

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
        self.model.eval()
        # self.evaluator1.reset()
        # self.evaluator2.reset()
        # self.evaluator3.reset()
        self.evaluator.reset()
        # self.evaluator5.reset()
        # self.evaluator6.reset()
        # self.evaluator7.reset()
        # self.evaluator8.reset()
        # self.evaluator9.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss1 = 0.0
        test_loss2 = 0.0
        test_loss3 = 0.0
        test_losscl=0.0
        test_loss = 0.0
        num_img_tr = len(self.val_loader)
        print(num_img_tr)
        for i, sample in enumerate(tbar):
            image, target, con0, con1, con2, con3, con4, con_d1_0, con_d1_1, con_d1_2, con_d1_3, con_d1_4  = \
                sample[0]['image'], sample[0]['label'], sample[0]['connect0'], sample[0]['connect1'], sample[0]['connect2'], sample[0]['connect3'], sample[0]['connect4'], \
                sample[0]['connect_d1_0'], sample[0]['connect_d1_1'], sample[0]['connect_d1_2'], sample[0]['connect_d1_3'], sample[0]['connect_d1_4']
            connect_label = torch.cat((con0[:,0,:,:].reshape(-1,1,512,512), con1[:,0,:,:].reshape(-1,1,512,512), con2[:,0,:,:].reshape(-1,1,512,512), con3[:,0,:,:].reshape(-1,1,512,512), con4[:,0,:,:].reshape(-1,1,512,512)), 1)
            connect_d1_label = torch.cat((con_d1_0[:,0,:,:].reshape(-1,1,512,512), con_d1_1[:,0,:,:].reshape(-1,1,512,512), con_d1_2[:,0,:,:].reshape(-1,1,512,512), con_d1_3[:,0,:,:].reshape(-1,1,512,512), con_d1_4[:,0,:,:].reshape(-1,1,512,512)), 1)

            if self.args.cuda:
                image, target, connect_label, connect_d1_label = image.cuda(), target.cuda(), connect_label.cuda(), connect_d1_label.cuda()
            with torch.no_grad():
                output, out_connect, out_connect_d1 = self.model(image)
            target = torch.unsqueeze(target, 1)
            loss1 = self.criterion(output, target)
            loss_cl=self.centralLoss(output,target)

            loss2 = self.criterion_con(out_connect, connect_label)
            loss3 = self.criterion_con(out_connect_d1, connect_d1_label)
            lad = 0.2
            loss = loss1 + lad * (0.6*loss2 + 0.4*loss3)+loss_cl
            test_loss1 += loss1.item()
            test_losscl+=loss_cl.item()
            test_loss2 += lad * 0.6 * loss2.item()
            test_loss3 += lad * 0.4 * loss3.item()
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f, loss1: %.6f, loss2: %.3f, loss3: %.3f, losscl: %.3f' % (test_loss / (i + 1), test_loss1 / (i + 1), test_loss2 / (i + 1), test_loss3 / (i + 1),test_losscl/(i+1)))
            pred = output.data.cpu().numpy()
            target_n = target.cpu().numpy()
            # Add batch sample into evaluator
            # pred1 = pred.copy()
            # pred2 = pred.copy()
            # pred3 = pred.copy()
            # pred5 = pred.copy()
            # pred6 = pred.copy()
            # pred7 = pred.copy()
            # pred8 = pred.copy()
            # pred9 = pred.copy()

            # pred1[pred1 > 0.1] = 1
            # pred1[pred1 < 0.1] = 0
            # self.evaluator1.add_batch(target_n, pred1)
            # pred2[pred2 > 0.2] = 1
            # pred2[pred2 < 0.2] = 0
            # self.evaluator2.add_batch(target_n, pred2)
            # pred3[pred3 > 0.3] = 1
            # pred3[pred3 < 0.3] = 0
            # self.evaluator3.add_batch(target_n, pred3)
            pred[pred > 0.4] = 1
            pred[pred < 0.4] = 0
            self.evaluator.add_batch(target_n, pred)
            # pred5[pred5 > 0.5] = 1
            # pred5[pred5 < 0.5] = 0
            # self.evaluator5.add_batch(target_n, pred5)
            # pred6[pred6 > 0.6] = 1
            # pred6[pred6 < 0.6] = 0
            # self.evaluator6.add_batch(target_n, pred6)
            # pred7[pred7 > 0.7] = 1
            # pred7[pred7 < 0.7] = 0
            # self.evaluator7.add_batch(target_n, pred7)
            # pred8[pred8 > 0.8] = 1
            # pred8[pred8 < 0.8] = 0
            # self.evaluator8.add_batch(target_n, pred8)
            # pred9[pred9 > 0.9] = 1
            # pred9[pred9 < 0.9] = 0
            # self.evaluator9.add_batch(target_n, pred9)

            # if i % (num_img_tr // 1) == 0:
            #     self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, i,
            #                                  split='Val')

        # Fast test during the training
        #Acc = self.evaluator.Pixel_Accuracy()
        #Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        IoU = self.evaluator.Intersection_over_Union()
        #Precision = self.evaluator.Pixel_Precision()
        #Recall = self.evaluator.Pixel_Recall()
        #F1 = self.evaluator.Pixel_F1()
        # dice1=self.evaluator1.Dice_coefficient()
        # dice2 = self.evaluator2.Dice_coefficient()
        # dice3 = self.evaluator3.Dice_coefficient()
        dice=self.evaluator.Dice_coefficient()
        # dice5 = self.evaluator5.Dice_coefficient()
        # dice6 = self.evaluator6.Dice_coefficient()
        # dice7 = self.evaluator7.Dice_coefficient()
        # dice8 = self.evaluator8.Dice_coefficient()
        # dice9 = self.evaluator9.Dice_coefficient()

        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/loss1_epoch', test_loss1/num_img_tr, epoch)
        self.writer.add_scalar('val/loss2_epoch', test_loss2, epoch)
        self.writer.add_scalar('val/loss3_epoch', test_loss3, epoch)
        self.writer.add_scalar('val/losscl_epoch', test_losscl, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        #self.writer.add_scalar('val/Acc', Acc, epoch)
        #self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/IoU', IoU, epoch)
        #self.writer.add_scalar('val/Precision', Precision, epoch)
        #self.writer.add_scalar('val/Recall', Recall, epoch)
        #self.writer.add_scalar('val/F1', F1, epoch)
        self.writer.add_scalar('val/Dice', dice, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("dice:{}"
              .format(dice))
        print('Loss: %.3f, Loss1: %.3f, Loss2: %.3f, Loss3: %.3f' % (test_loss, test_loss1, test_loss2, test_loss3))

        new_pred = dice
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
        else:
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

def main():
    parser = argparse.ArgumentParser(description="PyTorch CoANet Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=8,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='DeepGlobe',
                        choices=['spacenet', 'DeepGlobe'],
                        help='dataset name (default: spacenet)')
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=False,
                        help='whether to use sync bn')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='con_ce',
                        choices=['ce', 'con_ce', 'focal'],
                        help='loss func type')
    # training hyper params

    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                training (default: 16)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0,1,2,3)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point

    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    # parser.add_argument('--resume', type=str,
    #                     default="E:/jz/CoANet/rundscxy/DeepGlobe/CoANet-resnet/model_best.pth.tar",
    #                     help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')

    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='evaluuation interval (default: 10)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must bomma-separated list of integers only')

    import warnings
    warnings.filterwarnings("ignore")
    if args.checkname is None:
        args.checkname = 'CoANet-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        #trainer.validation(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1) and epoch>=100:
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
