import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
class Bpgan_VGGExtractor(nn.Module):
    def __init__(self,d):
        super(Bpgan_VGGExtractor, self).__init__()
        self.in_channel = int(d/40)
        self.freq_dim = 40
        self.out_dim = 1280

        self.conv1 = nn.Conv2d(self.in_channel, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(    64, 64, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2) # Half-time dimension
        self.conv3 = nn.Conv2d(    64,128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(   128,128, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2) # Half-time dimension
    def view_input(self,feature,xlen):
        # drop time
        xlen = [x//4 for x in xlen]
        if feature.shape[1]%4 != 0:
            feature = feature[:,:-(feature.shape[1]%4),:].contiguous()
        bs,ts,ds = feature.shape
        # reshape
        feature = feature.view(bs,ts,self.in_channel,self.freq_dim)
        feature = feature.transpose(1,2)

        return feature,xlen

    def forward(self,feature,xlen):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        feature,xlen = self.view_input(feature,xlen)
        feature = F.relu(self.conv1(feature))
        feature = F.relu(self.conv2(feature))
        feature = self.pool1(feature) # BSx64xT/2xD/2
        feature = F.relu(self.conv3(feature))
        feature = F.relu(self.conv4(feature))
        feature = self.pool2(feature) # BSx128xT/4xD/4
        # BSx128xT/4xD/4 -> BSxT/4x128xD/4
        feature = feature.transpose(1,2)
        #  BS x T/4 x 128 x D/4 -> BS x T/4 x 32D
        feature = feature.contiguous().view(feature.shape[0],feature.shape[1],self.out_dim)
        return feature,xlen

    def feature_map(self,feature):
        feature,xlen = self.view_input(feature,[128])
        feature = F.relu(self.conv1(feature))
        feature = F.relu(self.conv2(feature))
        out_feature1 = self.pool1(feature)
        feature = F.relu(self.conv3(out_feature1))
        feature = F.relu(self.conv4(feature))
        out_feature2 = self.pool2(feature)
        return out_feature1,out_feature2
    def load_param(self,path):
        self.load_state_dict(torch.load(path))

class Bpgan_VGGLoss(nn.Module):
    def __init__(self, d=40,sampling_ratio = 16000, n_fft=512, n_mels = 128,path = None):
        super(Bpgan_VGGLoss, self).__init__()
        self.vgg = Bpgan_VGGExtractor(d=d).cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/2, 1.0]
        A =  librosa.filters.mel(sr=sampling_ratio,n_fft=n_fft,n_mels=d)
        B = librosa.filters.mel(sr=sampling_ratio,n_fft=n_fft,n_mels=n_mels)
        C = A.dot(np.linalg.pinv(B))
        self.Transform_tensor = torch.Tensor(C).cuda()
        if path != None:
            self.vgg.load_param(path)
        else:
            if sampling_ratio == 16000:
                self.vgg.load_param("./models/VGG_Extractor_16k.pt")
            elif sampling_ratio == 8000:
                self.vgg.load_param("./models/VGG_Extractor_8k.pt")
            else:
                raise  NotImplementedError

    def forward(self, x, y):
        x_img =  torch.einsum("mj,idjk->idmk", [self.Transform_tensor, x])
        x_img = x_img[:, 0, :, :]
        x_img = x_img.transpose(1, 2)
        y_img = torch.einsum("mj,idjk->idmk", [self.Transform_tensor, y])
        y_img = y_img[:,0,:,:]
        y_img = y_img.transpose(1,2)
        x_vgg, y_vgg = self.vgg.feature_map(x_img), self.vgg.feature_map(y_img)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
    def load_param(self,path):
        self.vgg.load_param(path)
