
from datetime import datetime
import warnings
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter

from dataset.DaBRSTData import test_dataloader, train_dataloader,target_dataloader,targetvalidationloader
from models.FCN import  VGGNet,FCNs_240_3h
from models.discriminator import FCDiscriminator


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 5
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# def dice_loss(pred, target, smooth=1.):
#     pred = pred.contiguous()
#     target = target.contiguous()
#     prob = F.sigmoid(pred)
#
#     intersection = (prob * target).sum(dim=2).sum(dim=2)
#
#     loss = (1 - ((2. * intersection + smooth) / (prob.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
#
#     return loss.mean()



def caculateDICE(mri_vali_loader,  target_vali_loader ,net, visdom,smooth=1):
    print('开始进行MRI评分。。。')
    net.eval()
    print('开始进行CT评分。。。')
    total_CT_dice = 0
    for i, batch in tqdm(enumerate(target_vali_loader)):
        imgs = batch['image'].to(device=device, dtype=torch.float32)
        mask = batch['mask']
        predict, _,__,___ = net(imgs)
        prob = F.sigmoid(predict)
        prob = prob.cpu()
        prob = prob.detach()
        prob_np=(prob.numpy()>0.5)
        if i%30==0:
            show = (prob_np.copy()>0.5)*255
            label = mask.cpu().detach().numpy().copy()*255
            visdom.images(show, win='ct_test_pred{}'.format(i), opts=dict(title='test prediction'))
            visdom.images(label, win='ct_test_label{}'.format(i), opts=dict(title='test prediction'))
        prob_np=torch.from_numpy(prob_np)
        ct_msk_np = mask.cpu().detach().numpy().copy()
        ct_msk_np=torch.from_numpy(ct_msk_np)
        intersection = (prob_np * ct_msk_np).sum(dim=2).sum(dim=2)

        dice_coe = ((2. * intersection + smooth) / (prob_np.sum(dim=2).sum(dim=2) + ct_msk_np.sum(dim=2).sum(dim=2) + smooth))
        dice_coe = dice_coe.mean()
        del ct_msk_np ,prob_np
        total_CT_dice += dice_coe.item()
    mean_CT_dice = total_CT_dice / (i + 1)
    print('mean CT dice coe:', mean_CT_dice)
    mean_MRI_dice=0
    net.train()
    return mean_MRI_dice,mean_CT_dice

def caculateSensitivity(mri_vali_loader,  target_vali_loader ,net, visdom,smooth=1):
    print('starting evaluate MRI...')
    net.eval()
    print('starting evaluate CT...')
    total_CT_sensitivity = 0
    for i, batch in tqdm(enumerate(target_vali_loader)):
        imgs = batch['image'].to(device=device, dtype=torch.float32)
        mask = batch['mask']
        predict, _,__,___ = net(imgs)
        prob = F.sigmoid(predict)
        prob = prob.cpu()
        prob = prob.detach()
        prob_np=(prob.numpy()>0.5)
        prob_np=torch.from_numpy(prob_np)
        ct_msk_np = mask.cpu().detach().numpy().copy()
        ct_msk_np=torch.from_numpy(ct_msk_np)
        # intersection = (prob_np * ct_msk_np).sum(dim=2).sum(dim=2)
        tp=(prob_np * ct_msk_np).sum(dim=2).sum(dim=2)
        condition_positive=ct_msk_np.sum(dim=2).sum(dim=2)
        sensitivity=(tp+smooth)/(condition_positive+smooth)
        sensitivity = sensitivity.float().mean()
        del ct_msk_np, prob_np
        total_CT_sensitivity += sensitivity.item()
    mean_CT_sensitivity = total_CT_sensitivity/ (i + 1)
    print('mean CT sensitivity coe:', mean_CT_sensitivity)
    mean_MRI_dice=0
    net.train()
    return mean_MRI_dice,mean_CT_sensitivity


def caculateSpecificity(mri_vali_loader,  target_vali_loader ,net, visdom,smooth=1):
    print('开始进行MRI评分。。。')
    net.eval()
    print('开始进行CT评分。。。')
    total_CT_specificity = 0
    for i, batch in tqdm(enumerate(target_vali_loader)):
        imgs = batch['image'].to(device=device, dtype=torch.float32)
        mask = batch['mask']
        predict, _,__,___ = net(imgs)
        prob = F.sigmoid(predict)
        prob = prob.cpu()
        prob = prob.detach()
        prob_np=(prob.numpy()>0.5)
        prob_np=torch.from_numpy(prob_np)
        ct_msk_np = mask.cpu().detach().numpy().copy()
        allitem = ct_msk_np.size
        ct_msk_np=torch.from_numpy(ct_msk_np)
        # intersection = (prob_np * ct_msk_np).sum(dim=2).sum(dim=2)
        #tn
        real=(ct_msk_np).sum(dim=2).sum(dim=2)
        tn=allitem-real
        #pn
        pn=allitem-(prob_np).sum(dim=2).sum(dim=2)

        specificity=(pn+smooth)/(tn+smooth)
        specificity = specificity.float().mean()
        del ct_msk_np, prob_np
        total_CT_specificity += specificity.item()
    mean_CT_specificity = total_CT_specificity/ (i + 1)
    print('mean CT sensitivity coe:', mean_CT_specificity)
    mean_MRI_dice=0
    net.train()
    return mean_MRI_dice,mean_CT_specificity


source_label = 1
target_label = 0
warnings.filterwarnings('ignore')

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def unfix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

def fix_gn(m):
    classname = m.__class__.__name__
    if classname.find('GroupNorm') != -1:
        m.eval()

def train(epo_num=50, show_vgg_params=False):
    writer = SummaryWriter()
    total_step=0

    strength = 0.001
    vis = visdom.Visdom()
    # writer = SummaryWriter(comment=f'STRE_{strength}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    fcn_model = FCNs_240_3h(pretrained_net=vgg_model, n_class=1)

    fcn_model = fcn_model.to(device)
    d_model1 = FCDiscriminator(num_classes=32)
    d_model1.to(device)
    d_model2 = FCDiscriminator(num_classes=64)
    d_model2.to(device)
    d_model3 = FCDiscriminator(num_classes=128)
    d_model3.to(device)


    criterion = nn.BCELoss().to(device)
    bceloss = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-1, momentum=0.7)
    optimizer_D1 = optim.Adam(d_model1.parameters(), lr=0.001, betas=(0.9, 0.99))
    optimizer_D2 = optim.Adam(d_model2.parameters(), lr=0.001, betas=(0.9, 0.99))
    optimizer_D3 = optim.Adam(d_model2.parameters(), lr=0.001, betas=(0.9, 0.99))
    all_train_iter_loss = []

    # start timing
    prev_time = datetime.now()
    best_mri_dsc = 0
    best_ct_dsc=0
    best_ct_sensitivity = 0
    best_specificity=0
    for epo in range(epo_num):

        target_iter = enumerate(target_dataloader)
        train_loss = 0
        fcn_model.train()
        for index, batch in enumerate(train_dataloader):
            total_step+=1
            mri = batch['image']
            mri_msk = batch['mask']
            mri = mri.to(device).float()
            mri_msk = mri_msk.to(device).float()
            optimizer.zero_grad()
            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()
            optimizer_D3.zero_grad()
            # fcn_model.apply(fix_bn)
            output, h_mri,h_mri2,h_mri3 = fcn_model(mri)
            output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
            loss = criterion(output, mri_msk)
            loss.backward()

            _, twoctimgs = target_iter.__next__()
            ctimgs = twoctimgs[0].to(device=device, dtype=torch.float32)
            fcn_model.apply(fix_bn)
            seg1, h_ct,h_ct2,h_ct3 = fcn_model(ctimgs)
            fcn_model.apply(unfix_bn)
            fcn_model.train()
            seg1 = torch.sigmoid(seg1)
            ct_imgs_original = twoctimgs[1]

            for param in d_model1.parameters():
                param.requires_grad = False
            ct_out = d_model1(h_ct)
            slabel = torch.FloatTensor(ct_out.data.size()).fill_(source_label).to(device)
            advloss1 = bceloss(ct_out, slabel)
            advloss1 = strength * advloss1
            # advloss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            advloss1.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()


            for param in d_model1.parameters():
                param.requires_grad = True
            h_mri = h_mri.detach()
            h_ct = h_ct.detach()
            mri_out = d_model1(h_mri)
            slabel = torch.FloatTensor(mri_out.data.size()).fill_(source_label).to(device)
            L1 = bceloss(mri_out, slabel)
            ct_out = d_model1(h_ct)
            tlabel = torch.FloatTensor(ct_out.data.size()).fill_(target_label).to(device)
            L2 = bceloss(ct_out, tlabel)
            (L1+L2).backward()
            optimizer_D1.step()

            if index %375 == 0 and index > 0:
                mean_MRI_dice,mean_CT_dice = caculateDICE(mri_vali_loader=test_dataloader,target_vali_loader=targetvalidationloader,
                                                            visdom=vis,net=fcn_model)

                if mean_MRI_dice > best_mri_dsc:
                    state = {'net':fcn_model.state_dict(), 'optimizer':optimizer.state_dict()}
                    torch.save(state, 'checkpoints/DABRST_best_mri_last3h_{}.pt'.format(str(strength)))

                    best_mri_dsc = mean_MRI_dice

                if mean_CT_dice > best_ct_dsc:
                    state = {'net': fcn_model.state_dict(), 'optimizer': optimizer.state_dict()}
                    torch.save(state, 'checkpoints/jingxuan_DABRST_ori_best_ct_last3h_dice_{}.pt'.format(str(strength)))

                    best_ct_dsc = mean_CT_dice
                print('best MRI score：',best_mri_dsc)
                print('best CT score：', best_ct_dsc)
                writer.add_scalar('DSC_CT_score', mean_CT_dice, total_step)


            output_np = (output.cpu().detach().numpy().copy()>0.5)*255

            mri_msk_np = mri_msk.cpu().detach().numpy().copy()*255


            ct_np = ct_imgs_original.cpu().detach().numpy().copy()
            ct_out_np = (seg1.cpu().detach().numpy().copy()>0.5)*255


            if np.mod(index, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_dataloader), iter_loss))
                vis.images(output_np, win='mri_prediction!!!', opts=dict(title='mri prediction'))
                vis.images(mri_msk_np, win='mri_label!!', opts=dict(title='mri_label'))
                vis.images(ct_np, win='train_pred', opts=dict(title='ct_original'))
                vis.images(ct_out_np, win='train_label', opts=dict(title='ct_prediction'))
                vis.line(all_train_iter_loss, win='train_iter_loss', opts=dict(title='train iter loss'))

if __name__ == "__main__":


    train(epo_num=100, show_vgg_params=False)
