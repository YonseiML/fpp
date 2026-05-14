import argparse
import csv
import os
import time
from copy import deepcopy
from datetime import datetime

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from clip.custom_clip_iptp_bas import get_coop
from data.cls_to_names import *
from data.datautils import AugMixAugmenter, build_dataset
from data.fewshot_datasets import fewshot_datasets
from data.imagnet_prompts import imagenet_classes
from data.imagenet_variants import imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from utils.metrics import Calculator, accuracy
from utils.tools import Summary, AverageMeter, ProgressMeter, set_random_seed

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def reg_loss(text_feature, args):
    if 'otpt' in args.run_type:
        lambda_ = args.lambda_term
        
        Wwt  =  torch.matmul(text_feature,text_feature.T)
        wwt_norm_col_HT = torch.linalg.norm(Wwt,dim=-1)
        
        e = torch.eye(Wwt.shape[1], device=args.gpu)
        M_norm = torch.linalg.norm(Wwt, dim=0,keepdim=True)
        scaled_e = e * M_norm
        
        u = Wwt - scaled_e
        u_norm = torch.linalg.norm(u, dim=-1,keepdim=True)
        v = u/u_norm
        normalized_matrix_exp = v.unsqueeze(2)
        normalized_matrix_T_exp = v.unsqueeze(1)
        
        outer_products = normalized_matrix_exp @ normalized_matrix_T_exp
        divided_matrix = outer_products
        
        scaled_matrix = 2 * divided_matrix
        identity_matrix_dim = e.unsqueeze(0).expand(Wwt.shape[1], -1, -1)
        
        transformed_matrix = identity_matrix_dim - scaled_matrix
        Wwt_exp = Wwt.unsqueeze(2)
        
        Hx = torch.bmm(transformed_matrix, Wwt_exp) 
        Hx = Hx.squeeze(2)
        
        Ht_ortho = Hx - e  
        Ht_ortho_norm = torch.linalg.norm(Ht_ortho, dim=-1)
        loss_reg = Ht_ortho_norm.mean() * lambda_
        
    elif 'ctpt' in args.run_type:
        lambda_ = args.lambda_term
        
        text_feature_mean = text_feature.mean(0)
        l2_norm = torch.linalg.norm(text_feature - text_feature_mean, dim=-1)
        ATFD = l2_norm.mean()
        loss_reg = (-lambda_* ATFD)
        
    return loss_reg
            
def test_time_tuning(model, inputs, optimizer, scaler, args, cons, aux_outputs=None):
    output = None
    output2 = None
    selected_idx = None

    for j in range(args.tta_steps):
        if 'tpt' in args.run_type:
            with torch.cuda.amp.autocast():
                output = model(inputs,cons,args, aux_outputs) 

                if selected_idx is not None:
                    output = output[selected_idx]
                else:
                    output, selected_idx = select_confident_samples(output, args.selection_p)
                    softmax_out = torch.softmax(output,dim=-1)
                    
                
                loss_entropy = avg_entropy(output)
        else:
            loss_entropy = 0

        if 'otpt' in args.run_type:
            loss_orth = reg_loss(model.textfeatures_, args)        
            loss = loss_entropy + loss_orth 
              
        elif 'ctpt' in args.run_type:
            loss_var = reg_loss(model.textfeatures_, args)
            loss = loss_entropy + loss_var
            
        else: 
            loss = loss_entropy
            
            
        if args.run_type != 'baseline':
            optimizer.zero_grad()
            scaler.scale(loss).backward()
                    
            scaler.step(optimizer)
            scaler.update()

    return loss           

def main(args):

    set_random_seed(args.seed)

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)

def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # create model (zero-shot CLIP-ViT-B/16 with prompt tuning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes
    
    model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx,args.ctx_init,args.disp_cons)
    if args.load is not None:
        print("Use pre-trained soft prompt (CoOp) as initialization")
        pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
        assert pretrained_ctx.size()[0] == args.n_ctx
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(pretrained_ctx)
            model.prompt_learner.ctx_init_state = pretrained_ctx

    model_state = None

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    
    print("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define optimizer
    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, args.lr)
    optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    print('length of datset', len(datasets))
    for set_id in datasets:
        print('name id of dataset:', set_id)

    results = {}
    accuracy_data = {}
    ece_data = {}
    sce_data = {}
    aece_data = {}
    mce_data = {}
    aurc_data = {}

    for set_id in datasets:
        accuracy_data[set_id] = []
        ece_data[set_id] = []
        sce_data[set_id] = []
        aece_data[set_id] = []
        mce_data[set_id] = []
        aurc_data[set_id] = []

        if args.tpt:
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            
            if args.I_augmix:
                data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                            augmix=len(set_id)>=1)
            else:
                data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                            augmix=len(set_id)>1)
            batchsize = 1
        else:
            data_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = args.batch_size

        print("evaluating: {}".format(set_id))
        # Reset classnames of custom CLIP model
        if len(set_id) > 1:
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if set_id in ['A', 'R', 'V',]:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            
            else:
                classnames = classnames_all
                
        model.reset_classnames(classnames, args.arch)
            
        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
            
        results[set_id], result_dict = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, args.disp_cons, classnames, set_id)

        acc, ece, sce, aece, mce, aurc = Calculator(result_dict, args)
        accuracy_data[set_id].append(acc)
        ece_data[set_id].append(ece)
        sce_data[set_id].append(sce)
        aece_data[set_id].append(aece)
        mce_data[set_id].append(mce)
        aurc_data[set_id].append(aurc)
        
        try:
                    print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
        except:
                    print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))        

        del val_dataset, val_loader        
      
    print("======== Result Summary ========")

    dataset_ids = list(results.keys())

    file_path = args.csv_log  

    directory = os.path.dirname(file_path)

    custom_path = file_path

    os.makedirs(directory, exist_ok=True)
    
    file_exists = os.path.isfile(custom_path)
    
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("params: nstep	lr	bs")
    print("params: {}	    {}	{}".format(args.tta_steps, args.lr, args.batch_size))
    print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    for id in results.keys():
        print("{}".format(id), end="	")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id][0]), end="	")
    print("\n")

    #with open(file_path, 'a' if file_exists else 'w', newline='') as csvfile:
    with open(custom_path, 'a' if file_exists else 'w', newline='') as csvfile:    
        csvwriter = csv.writer(csvfile)

        if not file_exists:
            csvwriter.writerow(["params: nstep", "lr", "bs"])
            csvwriter.writerow([current_datetime, "params: {} {} {}".format(args.tta_steps, args.lr, args.batch_size)])
            csvwriter.writerow(["", "[set_id]", "Top-1 acc.", "Top-5 acc."])

        dataset_ids = list(results.keys())
        csvwriter.writerow(current_datetime)
        csvwriter.writerow([""] + dataset_ids)
        
        # Write the Top-1 accuracies
        top1_accs = ["Top-1 acc."] + ["{:.2f}".format(results[id][0]) for id in dataset_ids]
        csvwriter.writerow(top1_accs)

        # Write the Top-5 accuracies
        top5_accs = ["Top-5 acc."] + ["{:.2f}".format(results[id][1]) for id in dataset_ids]
        csvwriter.writerow(top5_accs)  

         # Write final accuracies
        final_acc = ["Accuracy"] + ["{:.2f}".format(accuracy_data[id][0]) for id in dataset_ids]
        csvwriter.writerow(final_acc)

        # Write the ECE
        ECE = ["ECE."] + ["{:.2f}".format(ece_data[id][0]) for id in dataset_ids]
        csvwriter.writerow(ECE)   
        
        # Write the SCE
        SCE = ["SCE."] + ["{:.4f}".format(sce_data[id][0]) for id in dataset_ids]
        csvwriter.writerow(SCE)
        
        # Write the AECE
        AECE = ["AECE."] + ["{:.2f}".format(aece_data[id][0]) for id in dataset_ids]
        csvwriter.writerow(AECE)
        
        # Write the MCE
        MCE = ["MCE."] + ["{:.2f}".format(mce_data[id][0]) for id in dataset_ids]
        csvwriter.writerow(MCE)
        
        # Write the AURC
        AURC = ["AURC."] + ["{:.6f}".format(aurc_data[id][0]) for id in dataset_ids]
        csvwriter.writerow(AURC)


def init_prompts(model, args, aux_outputs):
    ori_text_features = aux_outputs['text_features'].clone().detach()
    trainable_param   = model.prompt_learner.parameters()

    epochs = 1000
    optimizer = torch.optim.AdamW(trainable_param, 1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-4
    )
    scaler = torch.cuda.amp.GradScaler(init_scale=epochs)
     
    num_classes = ori_text_features.size(0)
    num_classes = torch.tensor(num_classes, dtype=torch.float32, device="cuda")
    n_ctx = args.n_ctx
    num_noise = args.num_noise
    if args.FL_lambda == -1:
        fl_weight = 1.0 + 15.0 / num_classes
    else:
        fl_weight = args.FL_lambda
    print("Weight for flatness loss:", fl_weight)
    
    # Get context dimension from prompt learner
    ctx_dim = model.prompt_learner.ctx_dim

    s = 0.141421
    random_ctx = torch.randn(n_ctx, ctx_dim, device='cuda', dtype=torch.float16) * s

    with torch.no_grad():
        if args.random_cls:
            model.prompt_learner.reset_random_classnames(n_cls=int(num_classes.item()), cls_length=2, arch=args.arch)
            ori_text_features = model.get_text_features(aux_outputs, norm=True)
            print("Randomly initializing class names")

        if args.predefined == 'random':
            model.prompt_learner.ctx.copy_(random_ctx)
            print("Randomly initializing prompt vectors")


    for step in range(epochs):
        text_features_noise = []
        with torch.no_grad():
            aux_outputs['aux_loss'] = []

            scale_for_ctx = args.ctx_noise
            scale_for_cls = args.cls_noise

            emb_dim = model.prompt_learner.ctx_dim
            
            sample_prompts = model.prompt_learner()
            seq_len = sample_prompts.shape[1]  # [n_cls, seq_len, emb_dim]
            
            suffix_len = seq_len - 1 - n_ctx
            
            noise_ctx = torch.randn(num_noise, n_ctx, emb_dim, device='cuda', dtype=torch.float16) * scale_for_ctx
            noise_sos = torch.randn(num_noise, 1, emb_dim, device='cuda', dtype=torch.float16) * scale_for_cls
            noise_suffix = torch.randn(num_noise, suffix_len, emb_dim, device='cuda', dtype=torch.float16) * scale_for_cls
           
            noise = torch.zeros(num_noise, seq_len, emb_dim, device='cuda', dtype=torch.float16)
            noise[:, 1:1+n_ctx, :] += noise_ctx
            noise[:, :1, :] += noise_sos
            noise[:, 1+n_ctx:, :] += noise_suffix
            noise = noise.unsqueeze(1) 
            
            text_features_noise = model.train_textfeatures(aux_outputs, noise)
            text_features_noise = text_features_noise.view(num_noise, text_features_noise.size(0)//num_noise, ori_text_features.size(-1))  # 4, C, 512
                
        aux_outputs['aux_loss'] = []
        text_features_wonoise = model.train_textfeatures(aux_outputs, None, norm=True)
        
        loss_alignment = torch.mean(
            (text_features_wonoise - ori_text_features) ** 2) * ori_text_features.size(-1)
        
        loss_flatness = -F.cosine_similarity(text_features_wonoise.unsqueeze(0), text_features_noise, dim=-1).mean() * fl_weight
        
        loss = loss_alignment + loss_flatness
        
        if step % 100 == 0:
            print("Step [{}/{}]: loss_alignment {:.4f}, loss_flatness {:.4f}".format(
                step, epochs, loss_alignment.item(), loss_flatness.item()
            ))
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()

    return model.prompt_learner.ctx

def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, cons,classnames,set_id):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    model.eval()
    with torch.no_grad():
        model.reset()
    end = time.time()

    softmax = torch.nn.Softmax(dim=1)

    result_dict = {'max_confidence': [], 'prediction': [], 'label': [], 'prob': []}

    aux_outputs = {}
    aux_outputs['mean_cosine'] = []
    aux_outputs['confidence'] = []
    aux_outputs['acc'] = []
    aux_outputs['prob'] = []

    if args.FPP:
        with torch.no_grad():
            model.reset()
            aux_outputs['aux_loss'] = []
            text_features = model.get_text_features(aux_outputs, norm=True)
        aux_outputs['text_features'] = text_features
        init_ctx = init_prompts(model, args, aux_outputs)
        aux_outputs['init_ctx'] = init_ctx.clone().detach()
        model.reset_classnames(classnames, args.arch)
    
    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        target = target.cuda(args.gpu, non_blocking=True)
        if args.tpt:
            images = torch.cat(images, dim=0)

        if 'otpt' in args.run_type or 'ctpt' in args.run_type:
            args.image = image
            
        if args.tta_steps > 0:
            with torch.no_grad():
                model.reset()
                if args.FPP:
                    pretrained_ctx = aux_outputs['init_ctx']
                    model.prompt_learner.ctx.copy_(pretrained_ctx)
        optimizer.load_state_dict(optim_state)

        test_time_tuning(model, images, optimizer, scaler, args, cons, aux_outputs=aux_outputs)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(image, cons, args, aux_outputs)

        softmax_output = softmax(output)

        max_confidence, max_index = torch.max(softmax_output, 1)

        result_dict['max_confidence'].append(max_confidence.item())
        result_dict['prediction'].append(max_index.item())
        result_dict['label'].append(target)
        result_dict['prob'].append(softmax_output.detach())
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)
    
    progress.display_summary()

    return [top1.avg, top5.avg], result_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--csv_log',type=str,help='path to save the CSV summary')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16', choices=['ViT-B/16'])
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=1, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained prompt embeddings')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--lambda_term' , type=float, default=0.0, help='lambda for c-tpt')
    parser.add_argument('--disp_cons' , type=int, nargs='+',default=[18.0],help='List of display constants')
    parser.add_argument('--run_type' , type=str, default='tpt', choices=['baseline', 'tpt', 'tpt_otpt', 'tpt_ctpt'])
    parser.add_argument('--I_augmix', action='store_true', default=False, help='augmix for I')

    # ------------------------ FPP ------------------------
    parser.add_argument('--FPP', action='store_true', default=False, help='FPP initialization')
    parser.add_argument('--predefined', type=str, default='random', choices=['ctx_init', 'random'])
    parser.add_argument('--random_cls', action='store_true', default=False, help='random class token initialization for FPP')
    parser.add_argument('--num_noise', type=int, default=4)
    parser.add_argument('--FL_lambda', default=-1, type=float, help='Weight for flatness loss')
    
    parser.add_argument('--ctx_noise', default=0.0707105, type=float)
    parser.add_argument('--cls_noise', default=0.141421, type=float)

    args = parser.parse_args()


    main(args)
