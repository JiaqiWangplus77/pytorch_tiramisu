import torch
import torch.nn as nn

from .layers_dilate import *


class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(4,4,4,4,4),
                 up_blocks=(4,4,4,4,4), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=12):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True,dilation=1))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])  # out: ModuleList() : empty
        self.transDownBlocks = nn.ModuleList([])
        
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i],upsample=0))
            cur_channels_count += (growth_rate*down_blocks[i])
            
            skip_connection_channel_counts.insert(0,cur_channels_count)

            self.transDownBlocks.append(TransitionDown(cur_channels_count))
#            print('--------------------------------------------------')
#            print('DenseBlock \n', self.denseBlocksDown[-1])
#            print('--------------------------------------------------')
#            print('cur_channels_count:', cur_channels_count)
#            print('--------------------------------------------------')
#            print('skip_connection_channel_counts:', skip_connection_channel_counts)
#            print('--------------------------------------------------')
#            print('transDownBlocks \n', self.transDownBlocks[-1])

        #####################
        #     Bottleneck    #
        #####################
        dilation = [1,1,2,3,4]
        self.add_module('bottleneck',Bottleneck(cur_channels_count,
                                     growth_rate, bottleneck_layers,dilation))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels
#        print('--------------------------------------------------')
#        print('prev_block_channels:', prev_block_channels)
#        print('cur_channels_count:', cur_channels_count)
        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        dilation = [1,1,1,1]
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels
            
#            print('--------------------------------------------------')
#            print('transUpBlocks \n', self.transUpBlocks[-1])
#            print('--------------------------------------------------')
#            print('cur_channels_count:', cur_channels_count)
#            print('--------------------------------------------------')
#            print('prev_block_channels:', prev_block_channels)
#            print('--------------------------------------------------')
#            print('denseBlocksUp \n', self.denseBlocksUp[-1])
        ## Final DenseBlock ##

        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

#        print('--------------------------------------------------')
#        print('transUpBlocks \n', self.transUpBlocks[-1])
#        print('--------------------------------------------------')
#        print('denseBlocksUp \n', self.denseBlocksUp[-1])

        ## Softmax ##

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
                   padding=0, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        #print('first convolutional layer: ',out.size())
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            #print(i, 'denseBlocksDown layer (appended): ',out.size())
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)
            #print(i, 'transDownBlocks layer: ',out.size())

        out = self.bottleneck(out)
        #print('bottleneck layer: ',out.size())
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            #print(i, 'to be skipped: ',skip.size())
            out = self.transUpBlocks[i](out, skip)
            #print(i, 'transUpBlocks layer: ',out.size())
            out = self.denseBlocksUp[i](out)
            #print(i, 'denseBlocksUp layer: ',out.size())

        out = self.finalConv(out)
        #print('final convolutional layer: ',out.size())
        #out = self.softmax(out)
        return out


def FCDenseNet57(n_classes,in_channels):
    return FCDenseNet(
        in_channels=3, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
        growth_rate=12, out_chans_first_conv=48, n_classes=n_classes)


def FCDenseNet67(n_classes,in_channels):
    return FCDenseNet(
        in_channels=3, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)


def FCDenseNet103(n_classes,in_channels):
    return FCDenseNet(
        in_channels=3, down_blocks=(4,5,7,10,12),
        up_blocks=(12,10,7,5,4), bottleneck_layers=15,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)
