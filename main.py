import argparse
import torch
import models
import data
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
from data.semi_data import SemiDataset
import math
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils import tensorboard
from models.Change_Detection import CD_Model
from torchvision import transforms
import torch.nn.functional as F
from copy import deepcopy
import random
import warnings


warnings.filterwarnings("ignore")


class Poly(_LRScheduler):
    def __init__(self, optimizer, num_epochs, iters_per_epoch, warmup_epochs=0, last_epoch=-1):
        self.iters_per_epoch = iters_per_epoch
        self.cur_iter = 0
        self.N = num_epochs * iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch
        super(Poly, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        T = self.last_epoch * self.iters_per_epoch + self.cur_iter
        factor = pow((1 - 1.0 * T / self.N), 0.9)
        if self.warmup_iters > 0 and T < self.warmup_iters:
            factor = 1.0 * T / self.warmup_iters

        self.cur_iter %= self.iters_per_epoch
        self.cur_iter += 1
        assert factor >= 0, 'error in lr_scheduler'
        return [base_lr * factor for base_lr in self.base_lrs]


def _get_available_devices(n_gpu):
    sys_gpu = torch.cuda.device_count()
    if sys_gpu == 0:
        n_gpu = 0
    elif n_gpu > sys_gpu:
        n_gpu = sys_gpu
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    available_gpus = list(range(n_gpu))
    return device, available_gpus


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return iu, np.nanmean(iu)


def batch_pix_accuracy(predict, target):

    # _, predict = torch.max(output, 1)

    predict = predict.int() + 1
    target = target.int() + 1

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target)*(target > 0)).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()


def txt2list(txt_path):
    # 功能：读取txt文件，并转化为list形式
    # txt_path：txt的路径；

    data_list = []
    with open(txt_path) as f:
        data = f.readlines()
    for line in data:
        line = line.strip("\n")  # 去除末尾的换行符
        data_list.append(line)

    # data_array = np.array(data_list)
    return data_list

# cp /home/chrisd/change/RC-Semi-Change/data/LEVIR/A/* /home/chrisd/change/RCSCD/data/SZTAKI-ext/A/

def parse_args():
    parser = argparse.ArgumentParser(description='RC Semi Change Detection Framework')
    # basic settings
    parser.add_argument('--save_dir', type=str, default='./saved/WHU/semi20-ext-levir/')
    parser.add_argument('--label_percent', type=int, choices=[5, 10, 20, 40, 100], default=20)
    parser.add_argument('--semi', type=bool, default=True)
    parser.add_argument('--data_dir', type=str, default='./data/SZTAKI-ext')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()
    return args


def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)

    valset = SemiDataset(args.data_dir, 'val', label_percent=args.label_percent)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    # <====================== Supervised training with labeled images (SupOnly) ======================>

    mode = 'train'
    trainset = SemiDataset(args.data_dir, mode, label_percent=args.label_percent)
    trainset.ids = 2 * trainset.ids if len(trainset.ids) < 200 else trainset.ids
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems()

    step1 = True
    if step1:
        if args.semi:
            best_model, checkpoints = train(mode, model, trainloader, valloader, optimizer, args)
            print('\n================> Total stage 1: '
                  'Supervised training on labeled images (SupOnly + CT_loss)')
        else:
            best_model = train_sup(mode, model, trainloader, valloader, optimizer, args)
            print('\n================> Total stage 1: '
                  'Supervised training on labeled images (SupOnly)')
            exit()

    # <===================================== Select Reliable IDs =====================================>
    print('\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training')
    step2 = True
    if step2:
        mode = 'select'
        dataset = SemiDataset(args.data_dir, mode, label_percent=args.label_percent)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

        # models = []
        # for i in range(3):
        #     model_i = deepcopy(model)
        #     model_i.module.load_state_dict(torch.load('./step_models/%d_model.pth' % i))
        #     models.append(model_i)
        # select_reliable(models, dataloader, args)

        select_reliable(checkpoints, dataloader, args)

    # <================================ Pseudo label reliable images =================================>
    print('\n\n\n================> Total stage 3/6: Pseudo labeling reliable images')
    step3 = True
    if step3:
        mode = 'label'
        dataset = SemiDataset(args.data_dir, mode, label_percent=args.label_percent)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

        # best_model = deepcopy(model)
        # best_model.module.load_state_dict(torch.load('./step_models/best_model.pth'))
        label(best_model, dataloader, args)

    # <================================== The 1st stage re-training ==================================>
    print('\n\n\n================> Total stage 4/6: The 1st stage re-training on labeled and reliable unlabeled images')
    step4 = True
    if step4:
        mode = 'semi'
        trainset = SemiDataset(args.data_dir, mode, label_percent=args.label_percent)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=16, drop_last=True)

        model, optimizer = init_basic_elems()

        final_model = train(mode, model, trainloader, valloader, optimizer, args)

        torch.save(final_model.module.state_dict(),
                   os.path.join(args.save_dir, f"{args.label_percent}_{'semi_model'}" + ".pth"))


def init_basic_elems():

    model = CD_Model(num_classes=2)
    # SETTING THE DEVICE
    device, availble_gpus = _get_available_devices(1)
    model = torch.nn.DataParallel(model, device_ids=availble_gpus)
    model.to(device)

    optimizer = SGD([{'params': filter(lambda p: p.requires_grad, model.module.get_other_params())},
                     {'params': filter(lambda p: p.requires_grad, model.module.get_backbone_params()),
                      'lr': 1e-2 / 10}], lr=1e-2, momentum=0.9, weight_decay=1e-4)

    return model, optimizer


def comtrastive_loss(pred, target, mean=False):
    output_pred = F.softmax(pred, dim=1)
    postive_pred = output_pred[:5]
    negtive_pred = output_pred[5:]
    M = target[:5].clone().float()

    loss_pos = F.mse_loss(postive_pred[:, 0, :, :], negtive_pred[:, 0, :, :], reduction='none') * (1 - M)
    loss_neg1 = F.mse_loss(postive_pred[:, 0, :, :], negtive_pred[:, 1, :, :], reduction='none') * M
    loss_neg2 = F.mse_loss(postive_pred[:, 1, :, :], negtive_pred[:, 0, :, :], reduction='none') * M

    loss_ct = loss_pos + loss_neg1 + loss_neg2
    if mean:
        loss_ct = loss_ct.mean()
    return loss_ct


def train_sup(mode, model, trainloader, valloader, optimizer, args):
    iters = 0
    previous_best = 0.0

    iter_per_epoch = len(trainloader)
    lr_scheduler = Poly(optimizer=optimizer, num_epochs=args.epochs, iters_per_epoch=iter_per_epoch)

    writer_dir = os.path.join(args.save_dir, mode)
    writer = tensorboard.SummaryWriter(writer_dir)

    for epoch in range(1, args.epochs + 1):
        print("\n==> Epoch %i, previous best = %.2f" % (epoch, previous_best))

        model.train()
        wrt_mode = 'train'
        total_loss = 0.0
        tbar = tqdm(trainloader)

        for i, (image_A, image_B, label, image_id) in enumerate(tbar):
            image_A, image_B, label = image_A.cuda(non_blocking=True), image_B.cuda(non_blocking=True), \
                                      label.cuda(non_blocking=True)
            optimizer.zero_grad()
            pred = model(image_A, image_B)
            loss = F.cross_entropy(input=pred, target=label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            iters += 1
            lr_scheduler.step(epoch=epoch - 1)
            tbar.set_description('CE_Loss: %.3f ' % (total_loss / (i + 1)))
            writer.add_scalar(f'{mode}/{wrt_mode}/CE_Loss', total_loss / (i + 1), iters)

        metric = meanIOU(num_classes=2)

        if epoch % 1 == 0:
            model.eval()
            wrt_mode = 'val'
            tbar = tqdm(valloader)
            total_correct, total_label = 0, 0
            with torch.no_grad():
                for image_A, image_B, label, image_id in tbar:
                    label, image_A, image_B = label.cuda(non_blocking=True), image_A.cuda(non_blocking=True), \
                                              image_B.cuda(non_blocking=True)
                    pred = model(image_A, image_B)
                    pred = torch.argmax(pred, dim=1)

                    metric.add_batch(pred.cpu().numpy(), label.cpu().numpy())
                    cIOU = metric.evaluate()[0][1]
                    correct, labeled = batch_pix_accuracy(pred, label)
                    total_correct, total_label = total_correct + correct, total_label + labeled

                    tbar.set_description('cIOU: %.2f PA: %.2f' %
                                         (cIOU * 100.0, (1.0 * total_correct / (np.spacing(1) + total_label))))
            cIOU *= 100.0
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)

            writer.add_scalar(f'{mode}/{wrt_mode}/cIOU', cIOU, epoch)
            writer.add_scalar(f'{mode}/{wrt_mode}/pixAcc', pixAcc, epoch)

            if cIOU > previous_best:
                if previous_best != 0:
                    os.remove(os.path.join(args.save_dir, '%.2f_best_model.pth' % previous_best))
                previous_best = cIOU
                torch.save(model.module.state_dict(), os.path.join(args.save_dir, '%.2f_best_model.pth' % cIOU))
                best_model = deepcopy(model)

    return best_model


def train(mode, model, trainloader, valloader, optimizer, args):
    iters = 0
    previous_best = 0.0

    if mode == 'train':
        checkpoints = []

    iter_per_epoch = len(trainloader)
    lr_scheduler = Poly(optimizer=optimizer, num_epochs=args.epochs, iters_per_epoch=iter_per_epoch)

    writer_dir = os.path.join(args.save_dir, mode)
    writer = tensorboard.SummaryWriter(writer_dir)

    for epoch in range(1, args.epochs + 1):
        print("\n==> Epoch %i, previous best = %.2f" % (epoch, previous_best))

        model.train()
        wrt_mode = 'train'
        total_loss = 0.0
        total_ce_loss = 0.0
        total_ct_loss = 0.0
        tbar = tqdm(trainloader)

        for i, (image_A, image_B, label, image_id) in enumerate(tbar):
            image_A, image_B, label = image_A.cuda(non_blocking=True), image_B.cuda(non_blocking=True), \
                                      label.cuda(non_blocking=True)
            optimizer.zero_grad()

            if random.random() < 0.5:
                A_l_aug = torch.cat((image_A, image_A), dim=0)
                B_l_aug = torch.cat((image_B, image_A), dim=0)
                target_l_aug = torch.cat((label, label * 0), dim=0)
            else:
                A_l_aug = torch.cat((image_A, image_B), dim=0)
                B_l_aug = torch.cat((image_B, image_B), dim=0)
                target_l_aug = torch.cat((label, label * 0), dim=0)

            pred = model(A_l_aug, B_l_aug)
            loss_ce = F.cross_entropy(input=pred, target=target_l_aug)
            loss_ct = comtrastive_loss(pred, target_l_aug, mean=True)
            loss = loss_ce + loss_ct

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce_loss += loss_ce.item()
            total_ct_loss += loss_ct.item()

            iters += 1
            lr_scheduler.step(epoch=epoch - 1)

            tbar.set_description('CE_Loss: %.3f CT_Loss: %.3f' % (total_ce_loss / (i + 1), total_ct_loss / (i + 1)))

            writer.add_scalar(f'{mode}/{wrt_mode}/CE_Loss', total_ce_loss / (i + 1), iters)
            writer.add_scalar(f'{mode}/{wrt_mode}/CT_Loss', total_ct_loss / (i + 1), iters)

            del image_A, image_B, label, pred, A_l_aug, B_l_aug, target_l_aug

        metric = meanIOU(num_classes=2)

        model.eval()
        wrt_mode = 'val'
        tbar = tqdm(valloader)
        total_correct, total_label = 0, 0

        with torch.no_grad():
            for image_A, image_B, label, image_id in tbar:
                label, image_A, image_B = label.cuda(non_blocking=True), image_A.cuda(non_blocking=True), \
                                          image_B.cuda(non_blocking=True)
                pred = model(image_A, image_B)
                pred = torch.argmax(pred, dim=1)

                metric.add_batch(pred.cpu().numpy(), label.cpu().numpy())
                cIOU = metric.evaluate()[0][1]

                correct, labeled = batch_pix_accuracy(pred, label)

                total_correct, total_label = total_correct + correct, total_label + labeled

                tbar.set_description('cIOU: %.2f PA: %.2f' %
                                     (cIOU * 100.0, (1.0 * total_correct / (np.spacing(1) + total_label))))

        cIOU *= 100.0
        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)

        writer.add_scalar(f'{mode}/{wrt_mode}/cIOU', cIOU, epoch)
        writer.add_scalar(f'{mode}/{wrt_mode}/pixAcc', pixAcc, epoch)

        del label, image_A, image_B, pred

        if cIOU > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(args.save_dir, '%.2f_best_model.pth' % previous_best))
            previous_best = cIOU
            torch.save(model.module.state_dict(), os.path.join(args.save_dir, '%.2f_best_model.pth' % cIOU))
            best_model = deepcopy(model)

        if mode == 'train' and (epoch in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
            checkpoints.append(deepcopy(model))
            if epoch == args.epochs:
                checkpoints.append(best_model)
            torch.save(model.module.state_dict(), os.path.join(args.save_dir, '%.2f_model.pth' % epoch))

    if mode == 'train':
        return best_model, checkpoints

    return best_model


def select_reliable(models, dataloader, args):

    for i in range(len(models)):
        models[i].eval()
    tbar = tqdm(dataloader)

    id_to_reliability = []

    with torch.no_grad():
        for image_A, image_B, label, image_id in tbar:
            label, image_A, image_B = label.cuda(non_blocking=True), image_A.cuda(non_blocking=True), \
                                      image_B.cuda(non_blocking=True)

            preds = []
            for model in models:
                preds.append(torch.argmax(model(image_A, image_B), dim=1).cpu().numpy())

            mIOU = []
            for i in range(len(preds) - 1):
                metric = meanIOU(num_classes=2)
                metric.add_batch(preds[i], preds[-1])
                # re_i = metric.evaluate()[0][1]
                # if np.isnan(re_i):
                #     mIOU.append(0.0)
                # else:
                #     mIOU.append(re_i)
                mIOU.append(metric.evaluate()[-1])

            reliability = sum(mIOU) / len(mIOU)
            id_to_reliability.append((image_id, reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    with open(os.path.join(args.data_dir, 'list', f"{args.label_percent}_{'reliable_ids'}" + ".txt"), 'w') as f:
        for elem in id_to_reliability[:len(id_to_reliability) // 2]:
            f.write(elem[0][0] + '\n')
    with open(os.path.join(args.data_dir, 'list', f"{args.label_percent}_{'unreliable_ids'}" + ".txt"), 'w') as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2:]:
            f.write(elem[0][0] + '\n')


def label(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)

    metric = meanIOU(num_classes=2)

    with torch.no_grad():
        for image_A, image_B, label, image_id in tbar:
            image_A, image_B = image_A.cuda(non_blocking=True), image_B.cuda(non_blocking=True)
            pred = model(image_A, image_B)
            pred = torch.argmax(pred, dim=1).cpu()

            metric.add_batch(pred.numpy(), label.numpy())
            mIOU = metric.evaluate()[0][1]
            # mIOU = metric.evaluate()[-1]

            pred[pred == 1] = 255
            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='L')
            pred.save(os.path.join(f"{args.data_dir}{'/pseudo_label'}_{args.label_percent}", os.path.basename(image_id[0])))

            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))
            del label, image_A, image_B, pred


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
