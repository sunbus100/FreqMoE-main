import torch
import numpy as np

def augmentation(augment_time):
    if augment_time == 'batch':
        return BatchAugmentation()
    elif augment_time == 'dataset':
        return DatasetAugmentation()


class BatchAugmentation():
    def __init__(self):
        pass

    # def freq_mask(self,x, y, rate=0.5, dim=1):
    #     xy = torch.cat([x,y],dim=1)
    #     xy_f = torch.fft.rfft(xy,dim=dim)
    #     m = torch.cuda.FloatTensor(xy_f.shape).uniform_() < rate
    #     freal = xy_f.real.masked_fill(m,0)
    #     fimag = xy_f.imag.masked_fill(m,0)
    #     xy_f = torch.complex(freal,fimag)
    #     xy = torch.fft.irfft(xy_f,dim=dim)
    #     return xy

    def flipping(self,x,y,rate=0):
        xy = torch.cat([x,y],dim=1)
        # print("flip")
        idxs = np.arange(xy.shape[1])
        idxs = list(idxs)[::-1]
        xy = xy[:,idxs,:]
        return xy

    def warping(self,x,y,rate=0):
        xy = torch.cat([x,y],dim=1)
        new_xy = torch.zeros_like(xy)
        for i in range(new_xy.shape[1]//2):
            new_xy[:,i*2,:] = xy[:,i + xy.shape[1]//2,:]
            new_xy[:,i*2 + 1,:] = xy[:,i + xy.shape[1]//2,:]
        return xy

    def noise(self,x,y,rate=0.05):
        xy = torch.cat([x,y],dim=1)
        noise_xy = (torch.rand(xy.shape)-0.5) * 0.1
        xy = xy + noise_xy.cuda()
        return xy

    def noise_input(self,x,y,rate=0.05):
        noise = (torch.rand(x.shape)-0.5) * 0.1
        x = x + noise.cuda()
        xy = torch.cat([x,y],dim=1)
        return xy

    def masking(self,x,y,rate=0.5):
        xy = torch.cat([x,y],dim=1)
        b_idx = np.arange(xy.shape[1])
        np.random.shuffle(b_idx)
        crop_num = int(xy.shape[1]*0.5)
        xy[:,b_idx[:crop_num],:] = 0
        return xy

    def masking_seg(self,x,y,rate=0.5):
        xy = torch.cat([x,y],dim=1)
        b_idx = int(np.random.rand(1)*xy.shape[1]//2)
        xy[:,b_idx:b_idx+xy.shape[1]//2,:] = 0
        return xy

    def freq_mask(self,x, y, rate=0.5, dim=1):
        xy = torch.cat([x,y],dim=1)
        xy_f = torch.fft.rfft(xy,dim=dim)
        m = torch.cuda.FloatTensor(xy_f.shape).uniform_() < rate
        amp = abs(xy_f)
        _, index = amp.sort(dim=dim, descending=True)
        dominant_mask = index > 5
        m = torch.bitwise_and(m,dominant_mask)
        freal = xy_f.real.masked_fill(m,0)
        fimag = xy_f.imag.masked_fill(m,0)
        xy_f = torch.complex(freal,fimag)
        xy = torch.fft.irfft(xy_f,dim=dim)
        return xy

    def freq_mask(self,x, y, rate=0.5, dim=1):
        xy = torch.cat([x,y],dim=1)
        xy_f = torch.fft.rfft(xy,dim=dim)
        m = torch.cuda.FloatTensor(xy_f.shape).uniform_() < rate
        amp = abs(xy_f)
        _, index = amp.sort(dim=dim, descending=True)
        dominant_mask = index > 5
        m = torch.bitwise_and(m,dominant_mask)
        freal = xy_f.real.masked_fill(m,0)
        fimag = xy_f.imag.masked_fill(m,0)
        xy_f = torch.complex(freal,fimag)
        xy = torch.fft.irfft(xy_f,dim=dim)
        return xy

    def freq_mix(self, x, y, rate=0.5, dim=1):
        xy = torch.cat([x,y],dim=dim)
        xy_f = torch.fft.rfft(xy,dim=dim)
        
        m = torch.cuda.FloatTensor(xy_f.shape).uniform_() < rate
        amp = abs(xy_f)
        _,index = amp.sort(dim=dim, descending=True)
        dominant_mask = index > 2
        m = torch.bitwise_and(m,dominant_mask)
        freal = xy_f.real.masked_fill(m,0)
        fimag = xy_f.imag.masked_fill(m,0)
        
        b_idx = np.arange(x.shape[0])
        np.random.shuffle(b_idx)
        x2, y2 = x[b_idx], y[b_idx]
        xy2 = torch.cat([x2,y2],dim=dim)
        xy2_f = torch.fft.rfft(xy2,dim=dim)

        m = torch.bitwise_not(m)
        freal2 = xy2_f.real.masked_fill(m,0)
        fimag2 = xy2_f.imag.masked_fill(m,0)

        freal += freal2
        fimag += fimag2

        xy_f = torch.complex(freal,fimag)
        
        xy = torch.fft.irfft(xy_f,dim=dim)
        return xy

class DatasetAugmentation():
    def __init__(self):
        pass

    def freq_dropout(self, x, y, dropout_rate=0.2, dim=0, keep_dominant=True):
        x, y = torch.from_numpy(x), torch.from_numpy(y)

        xy = torch.cat([x,y],dim=0)
        xy_f = torch.fft.rfft(xy,dim=0)

        m = torch.FloatTensor(xy_f.shape).uniform_() < dropout_rate

        freal = xy_f.real.masked_fill(m,0)
        fimag = xy_f.imag.masked_fill(m,0)
        xy_f = torch.complex(freal,fimag)
        xy = torch.fft.irfft(xy_f,dim=dim)

        x, y = xy[:x.shape[0],:].numpy(), xy[-y.shape[0]:,:].numpy()
        return x, y

    def freq_mix(self, x, y, x2, y2, dropout_rate=0.2):
        x, y = torch.from_numpy(x), torch.from_numpy(y)

        xy = torch.cat([x,y],dim=0)
        xy_f = torch.fft.rfft(xy,dim=0)
        m = torch.FloatTensor(xy_f.shape).uniform_() < dropout_rate
        amp = abs(xy_f)
        _,index = amp.sort(dim=0, descending=True)
        dominant_mask = index > 2
        m = torch.bitwise_and(m,dominant_mask)
        freal = xy_f.real.masked_fill(m,0)
        fimag = xy_f.imag.masked_fill(m,0)
        

        x2, y2 = torch.from_numpy(x2), torch.from_numpy(y2)
        xy2 = torch.cat([x2,y2],dim=0)
        xy2_f = torch.fft.rfft(xy2,dim=0)

        m = torch.bitwise_not(m)
        freal2 = xy2_f.real.masked_fill(m,0)
        fimag2 = xy2_f.imag.masked_fill(m,0)

        freal += freal2
        fimag += fimag2

        xy_f = torch.complex(freal,fimag)
        xy = torch.fft.irfft(xy_f,dim=0)
        x, y = xy[:x.shape[0],:].numpy(), xy[-y.shape[0]:,:].numpy()
        return x, y