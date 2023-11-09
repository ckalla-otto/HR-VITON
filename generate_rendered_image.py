import fire
import torch
from cp_dataset_test import get_agnostic
from test_generator import get_opt, remove_overlap, save_images, visualize_segmap, load_checkpoint_G
import torch.nn.functional as F
from networks import ConditionGenerator, load_checkpoint, make_grid
import torchgeometry as tgm
from torchvision.utils import save_image
from torchvision.utils import make_grid as make_image_grid
import torchvision.transforms as transforms
import os
import torch.nn as nn
from network_generator import SPADEGenerator
from PIL import Image
import numpy as np
import json

def save_tensor_to_image(img_tensor, img_path):
    tensor = (img_tensor.clone() + 1) * 0.5 * 255
    tensor = tensor.cpu().clamp(0, 255)

    try:
        array = tensor.numpy().astype('uint8')
    except:
        array = tensor.detach().numpy().astype('uint8')

    if array.shape[0] == 1:
        array = array.squeeze(0)
    elif array.shape[0] == 3:
        array = array.swapaxes(0, 1).swapaxes(1, 2)

    im = Image.fromarray(array)
    im.save(img_path, format='JPEG')
        
def load_preproc_images(input_image_folder:str):
    transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    im_name = os.path.join(input_image_folder,"person.jpg")
    c_name = "test_cloth"
    fine_width = 768
    fine_height = 1024
    semantic_nc = 13
    c = Image.open(os.path.join(input_image_folder,"cloth.jpg")).convert('RGB')
    c = transforms.Resize(fine_width, interpolation=2)(c)
    cm = Image.open(os.path.join(input_image_folder,"cloth_mask.jpg"))
    cm = transforms.Resize(fine_width, interpolation=0)(cm)

    c = transform(c)  # [-1,1]
    cm_array = np.array(cm)
    cm_array = (cm_array >= 128).astype(np.float32)
    cm = torch.from_numpy(cm_array)  # [0,1]
    cm.unsqueeze_(0)

    # person image
    im_pil_big = Image.open(im_name)
    im_pil = transforms.Resize(fine_width, interpolation=2)(im_pil_big)
    
    im = transform(im_pil)

    # load parsing image
    im_parse_pil_big = Image.open(os.path.join(input_image_folder,"parse.png"))
    im_parse_pil = transforms.Resize(fine_width, interpolation=0)(im_parse_pil_big)
    parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()
    im_parse = transform(im_parse_pil.convert('RGB'))
    
    labels = {
        0:  ['background',  [0, 10]],
        1:  ['hair',        [1, 2]],
        2:  ['face',        [4, 13]],
        3:  ['upper',       [5, 6, 7]],
        4:  ['bottom',      [9, 12]],
        5:  ['left_arm',    [14]],
        6:  ['right_arm',   [15]],
        7:  ['left_leg',    [16]],
        8:  ['right_leg',   [17]],
        9:  ['left_shoe',   [18]],
        10: ['right_shoe',  [19]],
        11: ['socks',       [8]],
        12: ['noise',       [3, 11]]
    }

    parse_map = torch.FloatTensor(20, fine_height, fine_width).zero_()
    parse_map = parse_map.scatter_(0, parse, 1.0)
    new_parse_map = torch.FloatTensor(semantic_nc, fine_height, fine_width).zero_()
    
    for i in range(len(labels)):
        for label in labels[i][1]:
            new_parse_map[i] += parse_map[label]
    
    parse_onehot = torch.FloatTensor(1, fine_height, fine_width).zero_()
    for i in range(len(labels)):
        for label in labels[i][1]:
            parse_onehot[0] += parse_map[label] * i

    # load image-parse-agnostic
    image_parse_agnostic = Image.open(os.path.join(input_image_folder,"parse_agnostic.png"))
    image_parse_agnostic = transforms.Resize(fine_width, interpolation=0)(image_parse_agnostic)
    parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
    image_parse_agnostic = transform(image_parse_agnostic.convert('RGB'))

    parse_agnostic_map = torch.FloatTensor(20, fine_height, fine_width).zero_()
    parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
    new_parse_agnostic_map = torch.FloatTensor(semantic_nc, fine_height, fine_width).zero_()
    for i in range(len(labels)):
        for label in labels[i][1]:
            new_parse_agnostic_map[i] += parse_agnostic_map[label]
            

    # parse cloth & parse cloth mask
    pcm = new_parse_map[3:4]
    im_c = im * pcm + (1 - pcm)
    
    # load pose points
    pose_map = Image.open(os.path.join(input_image_folder,"openpose_rendered.png"))
    pose_map = transforms.Resize(fine_width, interpolation=2)(pose_map)
    pose_map = transform(pose_map)  # [-1,1]
    
    with open(os.path.join(input_image_folder,"keypoints.json"), 'r') as f:
        pose_label = json.load(f)
        pose_data = pose_label['people'][0]['pose_keypoints_2d']
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 3))[:, :2]

    
    # load densepose
    densepose_map = Image.open(os.path.join(input_image_folder,"densepose.jpg"))
    densepose_map = transforms.Resize(fine_width, interpolation=2)(densepose_map)
    densepose_map = transform(densepose_map)  # [-1,1]
    agnostic = get_agnostic(im_pil_big, im_parse_pil_big, pose_data)
    agnostic = transforms.Resize(fine_width, interpolation=2)(agnostic)
    agnostic = transform(agnostic)
    

    result = {
        'c_name':   c_name,     # for visualization
        'im_name':  im_name,    # for visualization or ground truth
        # intput 1 (clothfloww)
        'cloth':    c,          # for input
        'cloth_mask':     cm,   # for input
        # intput 2 (segnet)
        'parse_agnostic': new_parse_agnostic_map,
        'densepose': densepose_map,
        'pose': pose_map,       # for conditioning
        # GT
        'parse_onehot' : parse_onehot,  # Cross Entropy
        'parse': new_parse_map, # GAN Loss real
        'pcm': pcm,             # L1 Loss & vis
        'parse_cloth': im_c,    # VGG Loss & vis
        # visualization
        'image':    im,         # for visualization
        'agnostic' : agnostic
        }
    
    return result


def add_batch_dimension(input_tensor:torch.Tensor):
    return input_tensor[None,:,:,:]

def main():
    opt = get_opt()
    inputs = load_preproc_images(opt.input_image_folder)
    input1_nc = 4  # cloth + cloth-mask
    input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
    tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
    tocg.cuda()
    tocg.eval()
    
       
    # generator
    opt.semantic_nc = 7
    generator = SPADEGenerator(opt, 3+3+3)
    generator.print_network()
    generator.eval()
    # Load Checkpoint
    load_checkpoint(tocg, opt.tocg_checkpoint,opt)
    load_checkpoint_G(generator, opt.gen_checkpoint,opt)
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss = gauss.cuda()
    
    with torch.no_grad():
        
        pose_map = add_batch_dimension(inputs['pose']).cuda()
        pre_clothes_mask = add_batch_dimension(inputs['cloth_mask']).cuda()
        label = add_batch_dimension(inputs['parse'])
        parse_agnostic = add_batch_dimension(inputs['parse_agnostic'])
        agnostic = add_batch_dimension(inputs['agnostic']).cuda()
        clothes = add_batch_dimension(inputs['cloth']).cuda() # target cloth
        densepose = add_batch_dimension(inputs['densepose']).cuda()
        im = add_batch_dimension(inputs['image'])
        input_label, input_parse_agnostic = label.cuda(), parse_agnostic.cuda()
        pre_clothes_mask = torch.FloatTensor((pre_clothes_mask.detach().cpu().numpy() > 0.5).astype(float)).cuda()

        # down
        pose_map_down = F.interpolate(pose_map, size=(256, 192), mode='bilinear')
        pre_clothes_mask_down = F.interpolate(pre_clothes_mask, size=(256, 192), mode='nearest')
        input_label_down = F.interpolate(input_label, size=(256, 192), mode='bilinear')
        input_parse_agnostic_down = F.interpolate(input_parse_agnostic, size=(256, 192), mode='nearest')
        agnostic_down = F.interpolate(agnostic, size=(256, 192), mode='nearest')
        clothes_down = F.interpolate(clothes, size=(256, 192), mode='bilinear')
        densepose_down = F.interpolate(densepose, size=(256, 192), mode='bilinear')

        shape = pre_clothes_mask.shape
        
        # multi-task inputs
        input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
        input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)

        # forward
        flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt,input1, input2)
        
        # warped cloth mask one hot

        warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(float)).cuda()


        if opt.clothmask_composition != 'no_composition':
            if opt.clothmask_composition == 'detach':
                cloth_mask = torch.ones_like(fake_segmap)
                cloth_mask[:,3:4, :, :] = warped_cm_onehot
                fake_segmap = fake_segmap * cloth_mask
                
            if opt.clothmask_composition == 'warp_grad':
                cloth_mask = torch.ones_like(fake_segmap)
                cloth_mask[:,3:4, :, :] = warped_clothmask_paired
                fake_segmap = fake_segmap * cloth_mask
                
        # make generator input parse map
        fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(opt.fine_height, opt.fine_width), mode='bilinear'))
        fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]


        old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_().cuda()

        old_parse.scatter_(1, fake_parse, 1.0)

        labels = {
            0:  ['background',  [0]],
            1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
            2:  ['upper',       [3]],
            3:  ['hair',        [1]],
            4:  ['left_arm',    [5]],
            5:  ['right_arm',   [6]],
            6:  ['noise',       [12]]
        }

        parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_().cuda()

        for i in range(len(labels)):
            for label in labels[i][1]:
                parse[:, i] += old_parse[:, label]
                
        # warped cloth
        N, _, iH, iW = clothes.shape
        flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
        flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)
        
        grid = make_grid(N, iH, iW,opt)
        warped_grid = grid + flow_norm
        warped_cloth = F.grid_sample(clothes, warped_grid, padding_mode='border')
        warped_clothmask = F.grid_sample(pre_clothes_mask, warped_grid, padding_mode='border')
        if opt.occlusion:
            warped_clothmask = remove_overlap(F.softmax(fake_parse_gauss, dim=1), warped_clothmask)
            warped_cloth = warped_cloth * warped_clothmask + torch.ones_like(warped_cloth) * (1-warped_clothmask)
        

        output = generator(torch.cat((agnostic, densepose, warped_cloth), dim=1), parse)
        save_tensor_to_image(output[0], "rendered_image.jpg")
    


if __name__ == "__main__":
    main()
