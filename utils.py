from dall_e         import map_pixels, unmap_pixels
import numpy        as np
import torchvision
import tempfile
import imageio
import random
import kornia
import shutil
import torch
import time
import os
import re


def create_outputfolder():
    outputfolder = os.path.join(os.getcwd(), 'output')
    if os.path.exists(outputfolder):
        shutil.rmtree(outputfolder)
    os.mkdir(outputfolder)

def create_strp(d, timeformat):
    return time.mktime(time.strptime(d, timeformat))

def download_stylegan_pt():
    cwd = os.getcwd()
    print(cwd)
    if 'stylegan.pt' not in os.listdir(cwd):
        url = "https://github.com/lernapparat/lernapparat/releases/download/v2019-02-01/karras2019stylegan-ffhq-1024x1024.for_g_all.pt"
        wget.download(url, "stylegan.pt")

def init_textfile(textfile):
    timeformat = "%M:%S"
    starttime = "00:00"
    with open(textfile, 'r') as file:
        descs = file.readlines()
        descs1 = [re.findall(r'(\d\d:\d\d) (.*)', d.strip('\n').strip())[0] for d in descs]
        if len(descs1[0]) == 0:
            descs1 = [re.findall(r'(\d\d:\d\d.\d\d) (.*)', d.strip('\n').strip())[0] for d in descs]
            timeformat = "%M:%S.%f"
            starttime = "00:00.00"
        descs1 = [(create_strp(d[0], timeformat), d[1])for d in descs1]
        firstline = (create_strp(starttime, timeformat), "start song")

        if descs1[0][0] - firstline[0]:
            descs1.insert(0, firstline)

        lastline = (descs1[-1][0]+9, "end song")
        descs1.append(lastline)
        
    return descs1

def create_image(img, i, text, gen, pre_scaled=True):
    if gen == 'stylegan':
        img = (img.clamp(-1, 1) + 1) / 2.0
        img = img[0].permute(1, 2, 0).detach().cpu().numpy() * 256
    else:
        img = np.array(img)[:,:,:]
        img = np.transpose(img, (1, 2, 0))
    if not pre_scaled:
        img = scale(img, 48*4, 32*4)
    img = np.array(img)
    with tempfile.NamedTemporaryFile() as image_temp:
        imageio.imwrite(image_temp.name+".png", img)
        image_temp.seek(0)
        return image_temp

nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

class Pars(torch.nn.Module):
    def __init__(self, gen='biggan'):
        super(Pars, self).__init__()
        self.gen = gen
        if self.gen == 'biggan':
            params1 = torch.zeros(32, 128).normal_(std=1).cuda()
            self.normu = torch.nn.Parameter(params1)
            params_other = torch.zeros(32, 1000).normal_(-3.9, .3)
            self.cls = torch.nn.Parameter(params_other)
            self.thrsh_lat = torch.tensor(1).cuda()
            self.thrsh_cls = torch.tensor(1.9).cuda()

        elif self.gen == 'dall-e':
            self.normu = torch.nn.Parameter(torch.zeros(1, 8192, 64, 64).cuda())

        elif self.gen == 'stylegan':
            latent_shape = (1, 1, 512)
            latents_init = torch.zeros(latent_shape).squeeze(-1).cuda()
            self.normu = torch.nn.Parameter(latents_init, requires_grad=True)

    def forward(self):
        if self.gen == 'biggan':
            return self.normu, torch.sigmoid(self.cls)

        elif self.gen == 'dall-e':
            # normu = torch.nn.functional.gumbel_softmax(self.normu.view(1, 8192, -1), dim=-1).view(1, 8192, 64, 64)
            normu = torch.nn.functional.gumbel_softmax(self.normu.view(1, 8192, -1), dim=-1, tau = 2).view(1, 8192, 64, 64)
            return normu


def pad_augs(image):
    pad = random.randint(1,50)
    pad_px = random.randint(10,90)/100
    pad_py = random.randint(10,90)/100
    pad_dims = (int(pad*pad_px), pad-int(pad*pad_px), int(pad*pad_py), pad-int(pad*pad_py))
    return torch.nn.functional.pad(image, pad_dims, "constant", 1)

def kornia_augs(image, sideX=512):
    blur = (random.randint(0,int(sideX/5))*2)+1
    kornia_model = torch.nn.Sequential(
        kornia.augmentation.RandomAffine(20, p=0.55, keepdim=True),
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.GaussianBlur((blur,blur),(blur,blur), p=0.5, border_type="constant"),
        kornia.augmentation.RandomSharpness(.5),
        kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.6)
    )
    return kornia_model(image)

def ascend_txt(model, lats, sideX, sideY, perceptor, percep, gen, tokenizedtxt):
    if gen == 'biggan':
        cutn = 128
        zs = [*lats()]
        out = model(zs[0], zs[1], 1)
        
    elif gen == 'dall-e':
        cutn = 32
        zs = lats()
        out = unmap_pixels(torch.sigmoid(model(zs)[:, :3].float()))

    elif gen == 'stylegan':
        zs = lats.normu.repeat(1,18,1)
        img = model(zs)
        img = torch.nn.functional.upsample_bilinear(img, (224, 224))

        img_logits, _text_logits = perceptor(img, tokenizedtxt.cuda())

        return 1/img_logits * 100, img, zs
    
    p_s = []
    for ch in range(cutn):
        # size = int(sideX*torch.zeros(1,).normal_(mean=.8, std=.3).clip(.5, .95))
        size = int(sideX*torch.zeros(1,).normal_(mean=.39, std=.865).clip(.362, .7099))
        offsetx = torch.randint(0, sideX - size, ())
        offsety = torch.randint(0, sideY - size, ())
        apper = out[:, :, offsetx:offsetx + size, offsety:offsety + size]
        apper = pad_augs(apper)
        # apper = kornia_augs(apper, sideX=sideX)
        apper = torch.nn.functional.interpolate(apper, (224, 224), mode='nearest')
        p_s.append(apper)
    into = torch.cat(p_s, 0)
    if gen == 'biggan':
        # into = nom((into + 1) / 2)
        up_noise = 0.01649
        into = into + (up_noise)*torch.randn_like(into, requires_grad=True)
        into = nom((into + 1) / 1.8)
    elif gen == 'dall-e':
        into = nom((into + 1) / 2)
    iii = perceptor.encode_image(into)
    llls = zs #lats()

    if gen == 'dall-e':
        return [0, 10*-torch.cosine_similarity(percep, iii).view(-1, 1).T.mean(1), zs]

    lat_l = torch.abs(1 - torch.std(llls[0], dim=1)).mean() + \
            torch.abs(torch.mean(llls[0])).mean() + \
            4*torch.max(torch.square(llls[0]).mean(), lats.thrsh_lat)
    for array in llls[0]:
        mean = torch.mean(array)
        diffs = array - mean
        var = torch.mean(torch.pow(diffs, 2.0))
        std = torch.pow(var, 0.5)
        zscores = diffs / std
        skews = torch.mean(torch.pow(zscores, 3.0))
        kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0
        lat_l = lat_l + torch.abs(kurtoses) / llls[0].shape[0] + torch.abs(skews) / llls[0].shape[0]
    cls_l = ((50*torch.topk(llls[1],largest=False,dim=1,k=999)[0])**2).mean()
    return [lat_l, cls_l, -100*torch.cosine_similarity(percep, iii, dim=-1).mean(), zs]

def train(i, model, lats, sideX, sideY, perceptor, percep, optimizer, text, tokenizedtxt, epochs=200, gen='biggan', img=None):
    loss1 = ascend_txt(model, lats, sideX, sideY, perceptor, percep, gen, tokenizedtxt)
    if gen == 'biggan':
        loss = loss1[0] + loss1[1] + loss1[2]
        zs = loss1[3]
    elif gen == 'dall-e':
        loss = loss1[0] + loss1[1]
        loss = loss.mean()
        zs = loss1[2]
    elif gen == 'stylegan':
        loss = loss1[0]
        img  = loss1[1].cpu()
        zs = loss1[2]


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i+1 == epochs:
        # if it's the last step, return the final z
        return zs
    return False
