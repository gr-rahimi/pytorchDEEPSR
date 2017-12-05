import torch
from torch import nn
from torchvision import models


class GeneratorBlock(nn.Module):
    def __init__(self, filters_count):
        super(GeneratorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(filters_count, filters_count, 3, padding = 1),
            nn.BatchNorm2d(filters_count),
            nn.PReLU(),
            nn.Conv2d(filters_count, filters_count,3, padding = 1),
            nn.BatchNorm2d(filters_count)
        )

    def forward(self, inp):
        return inp + self.block(inp)

class TwoXUpscaleBlock(nn.Module):
    def __init__(self, channel_count, filter_size):
        super(TwoXUpscaleBlock, self).__init__()
        self.block= nn.Sequential(
            nn.Conv2d(channel_count, 4 * channel_count, filter_size, padding = (filter_size - 1)/2),
            nn.PReLU(),
            nn.PixelShuffle(2)
        )
    def forward(self, inp):
        return self.block(inp)




class GeneratorNetwork(nn.Module):


    def __init__(self, log_up_scale_factor, residual_blocks_count):
        super(GeneratorNetwork, self).__init__()

        self.pre_block = nn.Sequential(
            nn.Conv2d(3,64,9, padding= 4 ),
            nn.PReLU())

        blocks_list = [GeneratorBlock(64) for _ in range(residual_blocks_count)]
        self.repeated_block = nn.Sequential(* blocks_list)

        self.last_conv_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.BatchNorm2d(64)
        )

        upscales_list = [TwoXUpscaleBlock(64, 3) for _ in range(log_up_scale_factor)]
        self.upscale_blocks = nn.Sequential(* upscales_list)

        self.last_conv = nn.Conv2d(64, 3, 9, padding = 4)



    def forward(self, inp):
        pre_block_output = self.pre_block(inp)
        repeated_block_output = self.repeated_block(pre_block_output)
        last_conv_block_output  = self.last_conv_block(repeated_block_output)
        upscale_block_output = self.upscale_blocks(last_conv_block_output)
        return  self.last_conv(upscale_block_output)



class DescriminatorBlock(nn.Module):
    def __init__(self, inp_filter, out_filter,kernel_size, stride):
        super(DescriminatorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inp_filter, out_filter, kernel_size, stride= stride, padding = (kernel_size-1)/2 ),
            nn.BatchNorm2d(out_filter),
            nn.LeakyReLU()
        )

    def forward(self, inp):
        return self.block(inp)

class DescriminatorNetwork(nn.Module) :
    def __init__(self, input_image_size=(112,112)):
        super(DescriminatorNetwork, self).__init__()
        self.pre_block = nn.Sequential(nn.Conv2d(3, 64, 3, padding = 1), nn.LeakyReLU())

        self.midlayer = nn.Sequential(
            DescriminatorBlock(64, 64, 3, 2),
            DescriminatorBlock(64, 128, 3, 1),
            DescriminatorBlock(128, 128, 3, 2),
            DescriminatorBlock(128, 256, 3, 1),
            DescriminatorBlock(256, 256, 3, 2),
            DescriminatorBlock(256, 512, 3, 1),
            DescriminatorBlock(512, 512, 3, 2)
        )

        conv_output_image_size = (input_image_size[0]/2**4,input_image_size[1]/2**4)
        self.last_layer = nn.Sequential(
            torch.nn.Linear(conv_output_image_size[0] * conv_output_image_size[1]*512, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024,1),
            torch.nn.Sigmoid()
        )

    def forward(self, inp):
        pre_block_output = self.pre_block(inp)
        mid_block_output = self.midlayer(pre_block_output)
        mid_block_output = mid_block_output.view(-1, 7*7*512 )
        last_layer_output = self.last_layer(mid_block_output)
        return last_layer_output


class VGG16_FE(nn.Module):
    def __init__(self):
        super(VGG16_FE, self).__init__()

        self.original_vgg16 = models.vgg16(pretrained = True) 

        self.features = nn.Sequential(
                *list(self.original_vgg16.features.children())[:29]
                )

	for param in self.features.parameters():
	    param.require_grad = False

    def forward(self, inp):
        return self.features(inp)












