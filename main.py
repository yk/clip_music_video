from utils          import train, Pars, create_image, create_outputfolder, init_textfile
from dall_e         import map_pixels, unmap_pixels, load_model
from stylegan       import g_synthesis
from biggan         import BigGAN
from tqdm           import tqdm

import create_video
import tempfile
import argparse
import torch
import clip
import glob
import os
import math


# Argsparse for commandline options
parser = argparse.ArgumentParser(description='BigGan_Clip')

parser.add_argument('--epochs', 
                    default = 100, 
                    type    = int, 
                    help    ='Number of Epochs')

parser.add_argument('--generator', 
                    default = 'biggan', 
                    type    = str, 
                    choices = ['biggan', 'dall-e', 'stylegan'],
                    help    = 'Choose what type of generator you would like to use BigGan or Dall-E')

parser.add_argument('--textfile', 
                    type    = str, 
                    required= True,
                    help    ='Path for the text file')

parser.add_argument('--audiofile', 
                    default = None,
                    type    = str, 
                    required= True,
                    help    ='Path for the mp3 file')

parser.add_argument('--lyrics', 
                    default = True,
                    type    = bool, 
                    help    ='Include lyrics')

parser.add_argument('--interpolation', 
                    default = 10,
                    type    = int, 
                    help    ='Number of elements to be interpolated per second and feed to the model')

args = parser.parse_args()

epochs      = args.epochs
generator   = args.generator
textfile    = args.textfile
audiofile   = args.audiofile
interpol    = args.interpolation
lyrics      = args.lyrics
sideX       = 512
sideY       = 512

def main():

    # Automatically creates 'output' folder
    create_outputfolder()

    # Initialize Clip
    perceptor, preprocess   = clip.load('ViT-B/32')
    perceptor               = perceptor.eval()

    # Load the model
    if generator == 'biggan':
        model   = BigGAN.from_pretrained('biggan-deep-512')
        model   = model.cuda().eval()
    elif generator == 'dall-e':
        model   = load_model("decoder.pkl", 'cuda')
    elif generator == 'stylegan':
        model   = g_synthesis.eval().cuda()

    # Read the textfile 
    # descs - list to append the Description and Timestamps
    descs = init_textfile(textfile)

    # list of temporary PTFiles 
    templist = []

    # Loop over the description list
    for d in tqdm(descs):

        timestamp = d[0]
        line = d[1]
        # stamps_descs_list.append((timestamp, line))

        lats = Pars(gen=generator).cuda()

         # Init Generator's latents
        if generator == 'biggan':
            par     = lats.parameters()
            lr      = 0.1#.07
        elif generator == 'stylegan':
            par     = [lats.normu]
            lr      = .01
        elif generator == 'dall-e':
            par     = [lats.normu]
            lr      = .1

        # Init optimizer
        optimizer = torch.optim.Adam(par, lr)

        # tokenize the current description with clip and encode the text
        txt = clip.tokenize(line)
        percep = perceptor.encode_text(txt.cuda()).detach().clone()

        # Training Loop
        for i in range(epochs):
            zs = train(i, model, lats, sideX, sideY, perceptor, percep, optimizer, line, txt, epochs=epochs, gen=generator)

        # save each line's last latent to a torch file temporarily
        latent_temp = tempfile.NamedTemporaryFile()
        torch.save(zs, latent_temp) #f'./output/pt_folder/{line}.pt')
        latent_temp.seek(0)
        #append it to templist so it can be accessed later
        templist.append(latent_temp)
    return templist, descs, model

def sigmoid(x):
    x = x * 2. - 1.
    return math.tanh(1.5*x/(math.sqrt(1.- math.pow(x, 2.)) + 1e-6)) / 2 + .5

def interpolate(templist, descs, model, audiofile):

    video_temp_list = []

    # interpole elements between each image

    for idx1, pt in enumerate(descs):

        # get the next index of the descs list, 
        # if it z1_idx is out of range, break the loop
        z1_idx = idx1 + 1
        if z1_idx >= len(descs):
            break

        current_lyric = pt[1]

        # get the interval betwee 2 lines/elements in seconds `ttime`
        d1 = pt[0]
        d2 = descs[z1_idx][0]
        ttime = d2 - d1

        # if it is the very first index, load the first pt temp file
        # if not assign the previous pt file (z1) to zs variable
        if idx1 == 0:
            zs = torch.load(templist[idx1])
        else:
            zs = z1
        
        # compute for the number of elements to be insert between the 2 elements
        N = round(ttime * interpol)
        print(z1_idx)
        # the codes below determine if the output is list (for biggan)
        # if not insert it into a list 
        if not isinstance(zs, list):
            z0 = [zs]
            z1 = [torch.load(templist[z1_idx])]
        else:
            z0 = zs
            z1 = torch.load(templist[z1_idx])
        
        # loop over the range of elements and generate the images
        image_temp_list = []
        for t in range(N):

            azs = []
            for r in zip(z0, z1):
                z_diff = r[1] - r[0] 
                inter_zs = r[0] + sigmoid(t / (N-1)) * z_diff
                azs.append(inter_zs)

            # Generate image
            with torch.no_grad():
                if generator == 'biggan':
                    img = model(azs[0], azs[1], 1).cpu().numpy()
                    img = img[0]
                elif generator == 'dall-e':
                    img = unmap_pixels(torch.sigmoid(model(azs[0])[:, :3]).cpu().float()).numpy()
                    img = img[0]
                elif generator == 'stylegan':
                    img = model(azs[0])
                image_temp = create_image(img, t, current_lyric, generator)
            image_temp_list.append(image_temp)

        video_temp = create_video.createvid(f'{current_lyric}', image_temp_list, duration=ttime / N)
        video_temp_list.append(video_temp)
    # Finally create the final output and save to output folder
    create_video.concatvids(descs, video_temp_list, audiofile, lyrics=lyrics)

if __name__ == '__main__':
    templist, descs, model = main()
    interpolate(templist, descs, model, audiofile)
