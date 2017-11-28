import argparse

from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms



parser = argparse.ArgumentParser()

parser.add_argument('--mode', type = str, default = 'train', help = 'train or test')
parser.add_argument('--train_path', type = str, default = './dataset/DIV2K_train_HR', help = 'path to train set')
parser.add_argument('--val_path', type = str, default = './dataset/DIV2K_valid_HR', help = 'path to test set')
parser.add_argument('--lr_crop_size', type = int, default = 28 , help = 'input low resoloution patch size')
parser.add_argument('--upscale_factor', type = int, default = 4, help = 'upscaling factor')
parser.add_argument('--use_cuda', help = 'use cuda')
parser.add_argument('--pre_train', help = 'pretrain generator network with MSE (No adverserial network)')
parser.add_argument('--pretrain_epochs', type = int, default = 50, help = 'number of pretrain epochs')
parser.add_argument('--batch_size', type = int, default = 32, help = 'training batch_size')

conf = parser.parse_args()




hr_to_random_patch_tensor = transforms.Compose([
    transforms.RandomCrop(conf.lr_crop_size * conf.upscale_factor),
    transforms.ToTensor()
])


def train():
    train_set = ImageFolder(root = conf.train_path, transform = hr_to_random_patch_tensor)
    train_loader = DataLoader(train_set, batch_size = parser.batch_size, shuffle=True,
                              num_workers = 4, pin_memory = conf.)
    print "train data loaded successfully from %s ()"%(conf.train_path)
    if conf.pre_train:
        print "Starting pre-training using MSE for "

        for i in xrange(conf.pre_train_epochs)

















