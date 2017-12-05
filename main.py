import argparse

from torch.utils.data import DataLoader
from torch.autograd import  Variable
import torch

from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.utils as vutils

import model
import math

from tensorboardX import SummaryWriter
import os


parser = argparse.ArgumentParser()

parser.add_argument('--mode', type = str, default = 'train', help = 'train or test')
parser.add_argument('--train_path', type = str, default = './dataset/DIV2K_train_HR', help = 'path to train set')
parser.add_argument('--val_path', type = str, default = './dataset/DIV2K_valid_HR', help = 'path to test set')
parser.add_argument('--lr_crop_size', type = int, default = 28 , help = 'input low resoloution patch size')
parser.add_argument('--upscale_factor', type = int, default = 4, help = 'upscaling factor')
parser.add_argument('--use_cuda', help = 'use cuda', action = 'store_true')
parser.add_argument('--pre_train', help = 'pretrain generator network with MSE (No adverserial network)', action = 'store_true')
parser.add_argument('--pretrain_epochs', type = int, default = 100, help = 'number of pretrain epochs')
parser.add_argument('--batch_size', type = int, default = 32, help = 'training batch_size')
parser.add_argument('--checkpoint_path', type = str, default = './ckpt', help = "check point directory" )
parser.add_argument('--train_epochs', type = int, default = 100 , help = "check point directory" )
parser.add_argument('--descriminator_pretrain_epochs', type = int, default = 50, help = "descriminator pretrain epochs" )
parser.add_argument('--load_pre_train', help = 'load pretrain G network', action = 'store_true')
parser.add_argument('--pre_train_d', help = 'pretrain descriminator network with pretrained MSE generator network', action = 'store_true')
parser.add_argument('--load_pre_train_d', help = 'load pretrain D network', action = 'store_true')
parser.add_argument('--val_crop_size', type = int, default = 400 , help = 'validation high resoloution patch size')
parser.add_argument('--g_pre_train_name', type = str , help = 'Generative pretrain network name')
parser.add_argument('--d_pre_train_name', type = str , help = 'Discriminative pretrained network name')


conf = parser.parse_args()

writer = SummaryWriter(conf.checkpoint_path)

hr_to_random_patch_tensor = transforms.Compose([
    transforms.RandomCrop(conf.lr_crop_size * conf.upscale_factor),
    transforms.ToTensor()
])

hr_to_center_patch_tensor = transforms.Compose([
    transforms.CenterCrop(conf.val_crop_size),
    transforms.ToTensor()
        ])

vgg_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale(224,interpolation = 3),
    transforms.ToTensor()
        ])

def get_resize_transform(dst_size):
    return transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale(dst_size, interpolation=3),
    transforms.ToTensor()
])

    

def cuda_factory(inp):
    if conf.use_cuda:
        return inp.cuda()
    else:
        return inp


def val(g_network, prefix_str):
    g_network.eval()
    val_set = ImageFolder(root = conf.val_path, transform = hr_to_center_patch_tensor)
    val_loader = DataLoader(val_set, batch_size = 8 , shuffle=False,
                                          num_workers = 4, pin_memory = False)
    mse_criterion = cuda_factory(torch.nn.MSELoss())
    mse_loss = 0.0
    for i , img_label_batch  in enumerate(val_loader):
        hr_batch, _ = img_label_batch
        hr_batch = Variable(hr_batch)
        lr_transformer = get_resize_transform(conf.val_crop_size/conf.upscale_factor )
        lr_batch = cuda_factory(Variable(torch.stack([lr_transformer(lr_img) - 0.5 for lr_img in hr_batch.data])))
        syn_hr_batch = g_network(lr_batch)
        mse_loss = mse_criterion(syn_hr_batch, hr_batch.cuda())
        if i == 0:
            imgs_to_show =[]
            resize_transform = get_resize_transform(conf.val_crop_size)
            lr_batch = lr_batch.cpu()
            hr_batch = hr_batch.cpu()
            syn_hr_batch = syn_hr_batch.cpu()
            for lr, hr, fake_hr in zip(lr_batch.data+0.5, hr_batch.data, syn_hr_batch.data):
                imgs_to_show.extend([resize_transform(lr), hr, fake_hr])

            imgs_to_show = torch.stack(imgs_to_show, 0)
            imgs_to_show = vutils.make_grid(imgs_to_show, nrow = 3, padding =5)

            writer.add_image(prefix_str+"images", imgs_to_show)

    g_network.train()

def train():

    train_set = ImageFolder(root = conf.train_path, transform = hr_to_random_patch_tensor)
    train_loader = DataLoader(train_set, batch_size = conf.batch_size, shuffle=True,
                              num_workers = 4, pin_memory = False)
    g_network = cuda_factory(model.GeneratorNetwork(2, 5)).train()
    g_network_optimizer = torch.optim.Adam(
        g_network.parameters(), lr = 1e-4
    )
    mse_criterion = cuda_factory(torch.nn.MSELoss())
    
    train_data_len = len(train_loader)
    print "train data loaded successfully from %s ()"%(conf.train_path)
    if conf.pre_train:
        print "Starting pre-training using MSE for "
        #mse_criterion = cuda_factory(torch.nn.MSELoss())
        for ep in xrange(conf.pretrain_epochs):
            pretrain_mse_loss= 0.0
            
            for i, img_label_batch in enumerate(train_loader):

                hr_batch, _ = img_label_batch
                hr_batch = Variable(hr_batch) # reuquires_grad = False?

                lr_transformer = get_resize_transform(conf.lr_crop_size)

                lr_batch = cuda_factory(Variable(torch.stack([lr_transformer(lr_img) - 0.5 for lr_img in hr_batch.data]))) # dim by default is zero :)
                hr_batch =cuda_factory(hr_batch)
                g_network.zero_grad()
                syn_hr_batch = g_network(lr_batch)
                mse_loss = mse_criterion(syn_hr_batch, hr_batch)
                mse_loss.backward()
                g_network_optimizer.step()
                pretrain_mse_loss += mse_loss.data[0]
                
                
                
            pretrain_mse_loss /= (train_data_len * conf.batch_size)
            print "Pretrain MSE: Epoch %d, MSE loss = %.4f, PSNR = %.4f"%(ep, pretrain_mse_loss,
                                                                         10 * math.log10(1/pretrain_mse_loss))

            writer.add_scalar('pretrain_MSE', pretrain_mse_loss, ep)
            writer.add_scalar('pretrain_PSNR', 10 * math.log10(1/pretrain_mse_loss), ep)
            val (g_network, 'pretrain_'+ str(ep))
        torch.save(g_network.state_dict(), os.path.join(conf.checkpoint_path, 'g_pretrain_mse.pth'))
        
    elif conf.load_pre_train:
        print 'Loading pretrained generator model...'
        g_network.load_state_dict(torch.load(os.path.join(conf.checkpoint_path, conf.g_pre_train_name)))

    d_network = cuda_factory(model.DescriminatorNetwork()).train()
    d_network_optimizer = torch.optim.Adam(d_network.parameters(), lr = 1e-4)
    criterion = cuda_factory(torch.nn.BCELoss())
    #label = cuda_factory(Variable(torch.FloatTensor(conf.batch_size)))
    if conf.pre_train_d:
        g_network.eval()
        for ep in xrange(conf.descriminator_pretrain_epochs):
            true_label,fake_label = 0.0, 0.0

            for i, img_label_batch in enumerate(train_loader):
                d_network.zero_grad()
                label = cuda_factory(Variable(torch.FloatTensor(img_label_batch[0].size(0))))
                hr_batch, _ = img_label_batch
                hr_batch = Variable(hr_batch)
                lr_transformer = get_resize_transform(conf.lr_crop_size)
                lr_batch = cuda_factory(Variable(torch.stack([lr_transformer(lr_img) - 0.5 for lr_img in hr_batch.data])))
                hr_batch = cuda_factory(hr_batch)
                label.data.fill_(1)
            
                real_hr_d_out = d_network(hr_batch).squeeze()
            
                errorD_real = criterion(real_hr_d_out, label)
                #print 'errorD_real', errorD_real.data
                errorD_real.backward()
            
                true_label += real_hr_d_out.data.mean()

            

                syn_hr_batch = g_network(lr_batch).detach()
                label.data.fill_(0)
                fake_hr_d_out = d_network(syn_hr_batch).squeeze()
            
                errorD_fake = criterion(fake_hr_d_out, label)
                #print 'errorD_fake = ', errorD_fake.data
                errorD_fake.backward()
                d_network_optimizer.step()
                fake_label += fake_hr_d_out.data.mean()

            true_label /= train_data_len
            fake_label /= train_data_len
        
            print 'Descriminator pretrain: Epoch %d, True images average label= %0.2f, fake images average label = %0.2f '%(ep,true_label,fake_label)
            writer.add_scalar('pretrain_D', (true_label + fake_label)/2, ep)

        torch.save(d_network.state_dict(), os.path.join(conf.checkpoint_path, 'd_pretrain.pth'))
    elif conf.load_pre_train_d:
        print 'Loading pretrained descriminator model...'
        d_network.load_state_dict(torch.load(os.path.join(conf.checkpoint_path, conf.d_pre_train_name)))


        
    g_network.train()
    d_network.train()
    g_network_optimizer = torch.optim.RMSprop(g_network.parameters(), lr = 1e-4)
    d_network_optimizer = torch.optim.RMSprop(d_network.parameters(), lr = 1e-4)
    vgg_net = cuda_factory(model.VGG16_FE())
    for ep in xrange(conf.train_epochs):
        true_label,fake_label, vgg_loss, mse_loss_log, ad_loss_log, total_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        for i , img_label_batch in enumerate(train_loader):
            
            d_network.zero_grad()
            label = cuda_factory(Variable(torch.FloatTensor(img_label_batch[0].size(0))))
            hr_batch, _ = img_label_batch
            hr_batch = Variable(hr_batch)
            lr_transformer = get_resize_transform(conf.lr_crop_size)
            lr_batch = cuda_factory(Variable(torch.stack([lr_transformer(lr_img) - 0.5 for lr_img in hr_batch.data])))
            hr_batch = cuda_factory(hr_batch)
            label.data.fill_(1)
            

            real_hr_d_out = d_network(hr_batch).squeeze()
            #print 'real data detection:',real_hr_d_out
            errorD_real = criterion(real_hr_d_out, label)
            errorD_real.backward()
            true_label += real_hr_d_out.data.mean()

            syn_hr_batch = g_network(lr_batch).detach()
            label.data.fill_(0)
            fake_hr_d_out = d_network(syn_hr_batch).squeeze()
            #print 'fake data detection:', fake_hr_d_out
            errorD_fake = criterion(fake_hr_d_out, label)
            errorD_fake.backward()
            d_network_optimizer.step()
            fake_label += fake_hr_d_out.data.mean()

            
            # Start training 
            g_network.zero_grad()
            syn_hr_batch = g_network(lr_batch)
            syn_hr_batch_cpu = syn_hr_batch.cpu()
            hr_batch_cpu = hr_batch.cpu()
            syn_hr_batch_224 = cuda_factory(Variable(torch.stack([vgg_transform(syn_hr_img) for syn_hr_img in syn_hr_batch_cpu.data])))
            hr_batch_224 = cuda_factory(Variable(torch.stack([vgg_transform(hr_img) for hr_img in hr_batch_cpu.data])))
            vgg16_syn_hr_batch = vgg_net(syn_hr_batch_224)
            vgg16_hr_batch = vgg_net(hr_batch_224)

            content_loss = mse_criterion(vgg16_syn_hr_batch, vgg16_hr_batch.detach())
            vgg_loss+=content_loss.data[0]

            mse_loss = mse_criterion(syn_hr_batch, hr_batch)
            mse_loss_log += mse_loss.data[0]

            d_output = d_network(syn_hr_batch).squeeze()
            label.data.fill_(1)
            
            ad_loss = criterion(d_output, label).mean()
            ad_loss_log += ad_loss.data[0]

            final_loss = mse_loss + 0.001 * content_loss + 0.0000 * ad_loss 
            total_loss += final_loss.data[0]

            final_loss.backward()

            g_network_optimizer.step()

        #print cont_loss
        true_label /= train_data_len
        fake_label /= train_data_len
        print 'epoch %d, final loss = %0.4f, real_detection = %0.4f, fake_detection= %0.4f '%(ep, total_loss, true_label, fake_label)
        val(g_network, 'train_'+str(ep))

        writer.add_scalar('train_D', (true_label + fake_label)/2, ep)
        writer.add_scalar('train_MSE', mse_loss_log, ep)
        writer.add_scalar('train_PSNR', 10 * math.log10(1/mse_loss_log), ep)
        writer.add_scalar('vgg_loss', vgg_loss, ep)
        writer.add_scalar('adv_loss', ad_loss_log, ep)
        writer.add_scalar('total_loss', total_loss, ep)

    torch.save(d_network.state_dict(), os.path.join(conf.checkpoint_path, 'd_train.pth'))
    torch.save(g_network.state_dict(), os.path.join(conf.checkpoint_path, 'g_train.pth'))












if __name__ == "__main__":
    if conf.mode == "train":
        train()

