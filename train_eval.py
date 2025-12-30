import os
os.environ['TIMM_FUSED_ATTN'] = '0' if 'TIMM_FUSED_ATTN' not in os.environ else os.environ['TIMM_FUSED_ATTN']
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # DEBUG
from time import time as ttime
import argparse
import random
from collections import OrderedDict
import tqdm
from typing import Any, Literal
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
import torchvision
import timm
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from utils.mod_adam import ModAdam
import utils.vit_builder
from utils.vit_builder import VisionTransformer
from utils.dataset_builder import ImagePathDatasetClassManager, ImagePathDataset, Mixup
from utils.continual_manager import ClassIncrementalManager
from utils import misc
from utils.dataset_utils import define_dataset
from utils.importance_utils import calc_importance_by_gradient, accumulate_importance_dict, select_important_params
from utils.nullspace_utils import get_interm_tensor_dict, accumulate_interm_tensor_dict, get_update_projection_dict
from utils.classifier_utils import ncm_classifier, extract_class_features, refine_head

torch.set_float32_matmul_precision("high")


class GlobalVarsManager:
    args: argparse.Namespace
    path_data_dict: dict[str, ImagePathDataset]
    cl_mngr: ClassIncrementalManager
    acc_mat_dict: OrderedDict[str, np.ndarray]
    cache_dict: dict
    param_dict: dict[Literal['base_params', 'task_params_'], OrderedDict[str, Tensor]]
    label_map_g2l: dict[int, tuple[int, int, int]]  # {original_label: (taskid, local_label, global_label)}

    def init_from_args(self, args):
        self.args = args
        _dataset_class_manager = ImagePathDatasetClassManager(**{args.dataset: args.data_root})
        self.path_data_dict = {'train': _dataset_class_manager[args.dataset](train=True),
                               'eval': _dataset_class_manager[args.dataset](train=False)}
        self.cl_mngr = ClassIncrementalManager(self.path_data_dict['eval'].class_list, args.num_tasks, args.seed, shuffle=args.shuffle_classes)
        self.acc_mat_dict = OrderedDict(AccTaskIncMat=np.zeros([_nt := self.cl_mngr.num_tasks, _nt]), AccClassIncMat=np.zeros([_nt, _nt]),
                                        AccTaskIncList=np.zeros([_nt := self.cl_mngr.num_tasks]), AccClassIncList=np.zeros([_nt]))
        self.cache_dict = {}
        self.param_dict = {}
        self.label_map_g2l = {}

    def update_label_maps(self, taskid: int, task_classes: list[int]) -> tuple[dict[int, int], dict[str, int]]:
        _g2l_map = misc.make_label_maps(taskid, task_classes)
        if not all([_k not in self.label_map_g2l.keys() for _k in _g2l_map.keys()]):
            print("The global_to_local label map has been fully loaded, which is not expected.")
        self.label_map_g2l.update(_g2l_map)
        return _g2l_map

def get_args():
    parser = argparse.ArgumentParser(description='Class-incremental Learning')
    # Experiment options
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=('cifar100', 'imagenet_r', 'sdomainet', 'eurosat', 'resisc', 'omni'), help='use lowercase')
    parser.add_argument('-dr', '--data_root', type=str, default="")
    parser.add_argument('-t', '--num_tasks', type=int, default=10, choices=(1, 2, 5, 9, 10, 20, 25, 50, 100))
    parser.add_argument('--shuffle_classes', type=misc.str2bool, default=True)
    parser.add_argument('--resume', type=str, default="")
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--seed', type=int, default=2025)
    # Model options
    parser.add_argument('-m', '--model', type=str, default='vit_base_patch16_224.augreg_in21k', help='vit_base_patch16_224.augreg_in21k')
    parser.add_argument('--head_dim_type', type=str, choices=('task_classes', 'pretrained'), default='pretrained')
    parser.add_argument('--logit_type', type=str, choices=('head_out',), default='head_out')
    parser.add_argument('--logit_scale', type=float, default=4.605170249938965, help='0 | 4.605170249938965')
    parser.add_argument('--prompt_len', type=int, default=0, help='0 means not using prompt')
    parser.add_argument('--prompt_init', type=str, choices=('uniform', 'zero', 'trunc_normal', 'xavier_uniform', 'kaiming_uniform'), default='uniform')
    parser.add_argument('--prompt_start_block', type=int, default=0, help='')
    parser.add_argument('--prompt_end_block', type=int, default=11, help="including both")
    parser.add_argument('--seperate_head', type=misc.str2bool, default=True)
    # Null space options
    parser.add_argument('--use_null_space', action='store_true')
    parser.add_argument('--null_patterns', type=str, nargs='+', default=('prompt',))
    parser.add_argument('--null_thres_mode', type=str, choices=('adaptive', 'times', 'num', 'pct', 'val'), default='adaptive')
    parser.add_argument('--null_thres_value1', type=float, default=0.)
    parser.add_argument('--null_thres_value2', type=float, default=0.)
    parser.add_argument('--null_alpha1', type=float, default=1.)
    parser.add_argument('--null_alpha2', type=float, default=1.)
    parser.add_argument('--null_interm_accum', type=str, default='sum')
    parser.add_argument('--null_interm_batches', type=int, default=-1, help="")
    parser.add_argument('--null_data_shuffle', type=misc.str2bool, default=False)
    parser.add_argument('--ln_loss_lam', type=float, default=1.)
    parser.add_argument('--ln_loss_tt', type=str, choices=('last', 'first'), default='last')
    parser.add_argument('--ln_loss_thres', type=float, default=0)
    # Importance computation options
    parser.add_argument('--impt_enable', type=misc.str2bool, default=True)
    parser.add_argument('--impt_batch_size', type=int, default=100)
    parser.add_argument('--impt_data_shuffle', type=misc.str2bool, default=False)
    parser.add_argument('--impt_use_tmpr', type=misc.str2bool, default=True)
    parser.add_argument('--impt_topk', type=float, default=0.05)
    parser.add_argument('--impt_loss_old', type=str, choices=('SupCon',), default='SupCon')
    parser.add_argument('--impt_loss_new', type=str, choices=('SupCon',), default='SupCon')
    parser.add_argument('--impt_supcon_repeat', type=int, default=1)
    parser.add_argument('--impt_contrast_augment', type=str, choices=('autoaug', 'original'), default='autoaug')
    parser.add_argument('--impt_norm_layer_grad', type=misc.str2bool, default=False)
    parser.add_argument('--impt_strategy', type=int, default=1)
    parser.add_argument('--impt_more_relax', type=float, default=0.03)
    parser.add_argument('--impt_momentum_old', type=float, default=1)
    parser.add_argument('--impt_lr_decay', type=float, default=1.)
    parser.add_argument('--impt_select_level', type=str, choices=('elem', 'token', 'dime'), default='dime')
    # Refine head options
    parser.add_argument('--refine_head', type=misc.str2bool, default=False)
    # Data augmentation options
    parser.add_argument('--transform_type', type=str, choices=('timm', 'autoaug', 'prototype'), default='autoaug')
    parser.add_argument('--interp_mode', type=str, choices=('auto', 'bilinear', 'bicubic'), default='auto')
    parser.add_argument('--prob_cutmixup', type=float, default=0)
    parser.add_argument('--cutmixup_stopepoch', type=int, default=999)
    # Training options
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-jt', '--workers', type=int, default=16)
    parser.add_argument('-je', '--eval_workers', type=int, default=2)
    parser.add_argument('-et', '--expand_times', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=28.)
    parser.add_argument('--use_amp', type=misc.str2bool, default=True)
    parser.add_argument('--use_compile', action='store_true')
    parser.add_argument('--sample_type', type=str, choices=('path', 'image'), default='image')
    parser.add_argument('--consecutive_training', type=misc.str2bool, default=True, help="")
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--persistent_workers', type=misc.str2bool, default=False)
    parser.add_argument('--training_string', type=str, nargs='+', default=('prompt', 'head'))
    parser.add_argument('-eb', '--eval_batch_size', type=int, default=100)
    parser.add_argument('--use_ncm', type=misc.str2bool, default=False)
    # Optimizer options
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('--lr_scale', type=float, default=1.)
    parser.add_argument('--lr_scale_patterns', type=str, nargs='+')
    parser.add_argument('--optimizer', type=str, default='mod_adam')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--lr_sch', type=str, default='multistep', choices=('cosine', 'step', 'multistep'))
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--min_lr', type=float, default=1e-5)  # for cosine and warmup
    parser.add_argument('-dm', '--decay_milestones', type=int, nargs='+', default=[5, 8])  # for multistep
    parser.add_argument('--decay_epochs', type=int, default=1000)  # for step
    parser.add_argument('--decay_rate', type=float, default=0.1)  # for step and multistep
    # Display options
    parser.add_argument('--show_bar', action='store_true')
    parser.add_argument('--print_model', action='store_true')

    args = parser.parse_args()
    if args.eval_only and not args.resume:
        raise AssertionError("Please provide 'resume' argument when 'eval_only' is True.")

    if args.interp_mode == 'auto':
        match args.dataset:
            case 'imagenet_r' | 'sdomainet' | 'omni':
                args.interp_mode = 'bilinear'
            case 'cifar100' | 'eurosat' | 'resisc':
                args.interp_mode = 'bicubic'
    if args.optimizer not in ('mod_adam',):
        raise NotImplementedError(args.optimizer)

    if not args.use_null_space:
        if args.ln_loss_lam != 0:
            print("WARNING:: args.ln_loss_lam is set to 0 for not using null space.")
            args.ln_loss_lam = 0

    if args.impt_select_level != 'elem':
        assert args.impt_topk == int(args.impt_topk) and args.impt_topk > 1, f"{args.impt_topk}"

    return args


def seed_etc_options(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.set_printoptions(precision=4, linewidth=256)
    torch.set_printoptions(linewidth=256)
    torchvision.set_image_backend('accimage')


def set_model_mode(GVM: GlobalVarsManager, model: VisionTransformer, training: bool, to_gpu: bool = True, training_string: tuple[str] = ('prompt',), ignore_pretrained_check: bool = False) -> VisionTransformer:
    for n, p in model.named_parameters():
        if training and any([_s in n for _s in training_string]):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    params_requires_grad = [n for n, p in model.named_parameters() if p.requires_grad]

    model.eval()
    for n, m in model.named_modules():
        # if training and any([_s == n for _s in training_string]):
        if training and any([n.endswith(_s) and not isinstance(m, nn.Identity) for _s in training_string]):
            m.train()
        else:
            m.eval()
    modules_training = [n for n, m in model.named_modules() if m.training]

    if to_gpu:
        model.cuda()

    if training:
        if not ignore_pretrained_check:
            for n in GVM.cache_dict['not_pretrained_params']:
                if n not in params_requires_grad:
                    raise ValueError(f"'{n}' does not require grad but it is in the 'not_pretrained_params' list: {GVM.cache_dict['not_pretrained_params']}.")
    else:
        assert len(params_requires_grad) == 0, f"{params_requires_grad}"
        assert len(modules_training) == 0, f"{modules_training}"

    return model


def set_learning_rates(GVM: GlobalVarsManager, model: VisionTransformer, base_lr: float, lr_scale: float, lr_scale_patterns: str) -> list[dict[str: Tensor | float]]:
    param_lr_groups = [{'params': [], 'lr': base_lr},
                       {'params': [], 'lr': base_lr * lr_scale}]
    lr_param_dict = {_p['lr']: [] for _p in param_lr_groups}

    for n, p in model.named_parameters():
        if p.requires_grad:
            _group_idx = 1 if any(_s in n for _s in lr_scale_patterns) else 0
            param_lr_groups[_group_idx]['params'].append(p)
            lr_param_dict[param_lr_groups[_group_idx]['lr']].append(n)

    return param_lr_groups


def train_one_epoch(GVM: GlobalVarsManager, curr_epoch: int, dataloader: DataLoader, model: VisionTransformer, criterion: nn.CrossEntropyLoss, optimizer: ModAdam) -> str:
    args = GVM.args
    temperature: float = args.temperature
    use_amp: bool = args.use_amp
    assert temperature > 0.
    if not args.use_null_space:
        assert args.ln_loss_lam == 0
    _use_cutmixup = args.prob_cutmixup > 0 and curr_epoch <= args.cutmixup_stopepoch

    if _use_cutmixup:
        cutmixup_fn = Mixup(mixup_alpha=1., cutmix_alpha=1., prob=args.prob_cutmixup, switch_prob=0.5, mode='batch', num_classes=len(GVM.cl_mngr.current_task_classes))

    amp_scalar = GradScaler(enabled=use_amp)
    scalar_meter = misc.ScalarMeter(lr="step_last:.3e", ce_loss="samp_avg:.4f", LN_mean_loss="samp_avg:.6f", LN_std_loss="samp_avg:.6f",
                                    loss="samp_avg:.4f", data_time="step_sum:.3f", batch_time="step_sum:.3f", acc_top1="samp_avg:>6.2%", acc_topk="samp_avg:>6.2%")
    _btimer = ttime()

    for i_batch, (images, target) in tqdm.tqdm(enumerate(dataloader, 1), total=len(dataloader), dynamic_ncols=True, disable=not GVM.args.show_bar):
        data_time = ttime() - _btimer

        images: Tensor = images.cuda(non_blocking=True)
        target: Tensor = target.cuda(non_blocking=True)

        if _use_cutmixup:
            mix_img, mix_lbl = cutmixup_fn(images, target)
        else:
            mix_img = images
            mix_lbl = target

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            logits: Tensor = model(mix_img)

        if i_batch == 1:
            if args.seperate_head:
                assert logits.shape[1] == len(GVM.cl_mngr.current_task_classes)
            else:
                assert logits.shape[1] == len(GVM.cl_mngr.sofar_task_classes)

        ce_loss = criterion(logits / temperature, mix_lbl)

        LN_mean_loss = torch.zeros_like(ce_loss)
        LN_std_loss = torch.zeros_like(ce_loss)
        if GVM.cl_mngr.current_taskid > 0:
            _dst_tt = GVM.cl_mngr.current_taskid - 1 if args.ln_loss_tt == 'last' else 0
            for _n0, _p0 in GVM.param_dict[f'task_params_{_dst_tt}'].items():
                if 'prompt' in _n0:
                    _p0 = _p0.detach()
                    _pt = model.get_parameter(_n0)
                    _mpt, _mp0 = _pt.mean(-1), _p0.mean(-1)
                    LN_mean_loss += _l if (_l := F.l1_loss(_mpt, _mp0)) > args.ln_loss_thres else 0
                    _spt, _sp0 = _pt.std(-1, unbiased=False), _p0.std(-1, unbiased=False)
                    LN_std_loss += _l if (_l := F.l1_loss(_spt, _sp0)) > args.ln_loss_thres else 0
                pass

        loss: Tensor = ce_loss + (LN_mean_loss + LN_std_loss) * args.ln_loss_lam

        optimizer.zero_grad()
        amp_scalar.scale(loss).backward()
        amp_scalar.step(optimizer)
        amp_scalar.update()

        acc_top1, acc_topk = misc.calc_accuracy(logits, target, topk=(1, 2))
        batch_time = ttime() - _btimer

        scalar_meter.add_step_value(len(images), lr=optimizer.param_groups[0]['lr'], ce_loss=ce_loss.item(), LN_mean_loss=LN_mean_loss.item(), LN_std_loss=LN_std_loss.item(),
                                    loss=loss.item(), data_time=data_time, batch_time=batch_time, acc_top1=acc_top1, acc_topk=acc_topk)
        _btimer = ttime()
        break # DEBUG

    _epoch_scalar_str = scalar_meter.format_outout(scalar_meter.update_epoch_average_value())
    return _epoch_scalar_str


def _save_chekpoint(GVM: GlobalVarsManager, taskid: int, epoch: int, model: VisionTransformer, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler):
    if taskid == 0:
        base_params = OrderedDict()
    task_params = OrderedDict()

    for n, p in model.named_parameters():
        if p.requires_grad:
            task_params[n] = p.clone()
        else:
            if taskid == 0:
                base_params[n] = p.clone()
            else:
                if not torch.all((_pb := GVM.param_dict[f'base_params'][n]) == p.to(_pb.device)):
                    print(f"WARNING:: save_chekpoint(): 'base_params' is changed!")

    _task_str = f"Task [{taskid + 1:>{len(_ntstr := str(GVM.cl_mngr.num_tasks))}}/{_ntstr}]"
    if taskid == 0:
        assert not 'base_params' in GVM.param_dict
        GVM.param_dict['base_params'] = base_params

    assert not f'task_params_{taskid}' in GVM.param_dict
    GVM.param_dict[f'task_params_{taskid}'] = task_params


def train_one_task(GVM: GlobalVarsManager, taskid: int, task_classes: list[int], model: VisionTransformer, **kwargs) -> VisionTransformer:
    args = GVM.args
    if args.epochs == 0:
        if args.use_ncm:
            extract_class_features(GVM, model)
        return model

    # Start training for one task
    print(f"*" * 90 + " Start Training " + "*" * 90)
    _ttimer = ttime()
    _ntstr = str(GVM.cl_mngr.num_tasks)

    model: VisionTransformer = set_model_mode(GVM, model, training=True, training_string=GVM.cache_dict['training_string'])
    model = modify_head(GVM, model, training=True, task_classes=task_classes)

    dataset = define_dataset(GVM, task_classes, training=True, transform_type=args.transform_type, target_map_to_local=args.seperate_head, expand_times=args.expand_times)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, timeout=args.timeout if args.workers > 0 else 0,
                            drop_last=args.prob_cutmixup > 0, persistent_workers=args.persistent_workers)

    criterion = nn.CrossEntropyLoss().cuda()

    if args.lr_scale == 1:
        param_groups = filter(lambda p: p.requires_grad, model.parameters())
    else:
        param_groups = set_learning_rates(GVM, model, args.lr, args.lr_scale, args.lr_scale_patterns)

    if taskid == 0:
        GVM.cache_dict['update_proj_dict'] = {}
    if args.use_null_space:
        if taskid == 0:
            GVM.cache_dict['null_param_id_dict'] = get_param_id_dict(model, args.null_patterns)
            GVM.cache_dict['interm_tensor_dict'] = {}
        else:
            assert list(GVM.cache_dict['update_proj_dict'].keys()) == list(GVM.cache_dict['null_param_id_dict'].keys())
    else:
        assert GVM.cache_dict['update_proj_dict'] == {}

    if args.optimizer == 'mod_adam':
        optimizer = ModAdam(param_groups, GVM.cache_dict['update_proj_dict'],
                            arg_dict={'mask_dict': GVM.cache_dict['mask_dict'], 'null_alpha': args.null_alpha1, 'impt_strategy': args.impt_strategy,
                                      'impt_more_relax': args.impt_more_relax, 'impt_lr_decay': args.impt_lr_decay} if taskid > 0 and args.impt_enable else None,
                            lr=args.lr, weight_decay=args.weight_decay, foreach=True)
    else:
        optimizer = create_optimizer_v2(param_groups, opt=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, foreach=True)

    scheduler, num_epochs = create_scheduler_v2(optimizer, sched=args.lr_sch, num_epochs=args.epochs, decay_epochs=args.decay_epochs, decay_milestones=args.decay_milestones,
                                                decay_rate=args.decay_rate, min_lr=args.min_lr, warmup_epochs=args.warmup_epochs, warmup_lr=args.min_lr)
    assert num_epochs == args.epochs

    # Statrt the training epochs
    torch.cuda.empty_cache()
    for epoch in range(0, args.epochs + 1):
        if epoch > 0:
            _epoch_scalar_str = train_one_epoch(GVM, epoch, dataloader, model, criterion, optimizer)
            print(f"Task [{taskid + 1:>{len(_ntstr)}}/{_ntstr}] Epoch [{epoch:>{len(_nestr := str(args.epochs))}}/{_nestr}]:: {_epoch_scalar_str}")
        scheduler.step(epoch)

    # End training for the task
    _save_chekpoint(GVM, taskid, epoch, model, optimizer, scheduler)

    if args.use_null_space and taskid + 1 < GVM.cl_mngr.num_tasks:
        new_interm_tensor_dict = get_interm_tensor_dict(GVM, model, GVM.cache_dict['null_param_id_dict'])
        GVM.cache_dict['interm_tensor_dict'] = accumulate_interm_tensor_dict(GVM, GVM.cache_dict['interm_tensor_dict'], new_interm_tensor_dict)
        GVM.cache_dict['update_proj_dict'] = get_update_projection_dict(GVM, GVM.cache_dict['null_param_id_dict'], GVM.cache_dict['interm_tensor_dict'])

    if args.refine_head or args.use_ncm:
        extract_class_features(GVM, model)
        if args.refine_head:
            refine_head(GVM, model)

    model.remove_text_features()

    print(f"Task [{taskid + 1:>{len(_ntstr)}}/{_ntstr}]:: Training time = {misc.format_duration(ttime() - _ttimer)}")

    return model


def evaluate_one_task(GVM: GlobalVarsManager, train_taskid: int, eval_taskid: int, eval_task_classes: list[int], model: VisionTransformer) -> OrderedDict[str, float]:
    use_amp: bool = GVM.args.use_amp
    _ttimer = ttime()

    dataset = define_dataset(GVM, eval_task_classes, training=False, transform_type=GVM.args.transform_type, target_map_to_local=False)
    dataloader = DataLoader(dataset, batch_size=GVM.args.eval_batch_size, shuffle=False, num_workers=GVM.args.eval_workers, pin_memory=True, timeout=GVM.args.timeout if GVM.args.eval_workers > 0 else 0)

    set_model_mode(GVM, model, training=False)
    scalar_meter = misc.ScalarMeter(acc_task_inc="samp_avg:>6.2%", acc_class_inc="samp_avg:>6.2%")

    for images, target in tqdm.tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, disable=not GVM.args.show_bar):
        images: Tensor = images.cuda(non_blocking=True)
        target: Tensor = target.cuda(non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                if GVM.args.use_ncm:
                    _feats: Tensor = model.encode_image(images, pre_logits=True)
                    logits = ncm_classifier(GVM, _feats)
                else:
                    logits: Tensor = model(images)

        assert logits.ndim == 2
        assert logits.shape[1] == len(GVM.cl_mngr.sofar_task_classes), f"{logits.shape}, {len(GVM.cl_mngr.sofar_task_classes)}"

        class_inc_preds = logits.argmax(dim=1)
        _task_inc_logits = logits.clone()
        _task_inc_logits[:, :eval_taskid * GVM.cl_mngr.num_classes_per_task] = -torch.inf
        _task_inc_logits[:, (eval_taskid + 1) * GVM.cl_mngr.num_classes_per_task:] = -torch.inf
        task_inc_preds = _task_inc_logits.argmax(dim=1)

        acc_task_inc, acc_topnn_task, num_nn_task = misc.calc_acc_topnn_dynamically(task_inc_preds, target)
        acc_class_inc, acc_topnn_class, num_nn_class = misc.calc_acc_topnn_dynamically(class_inc_preds, target)
        scalar_meter.add_step_value(target.shape[0], acc_task_inc=acc_task_inc, acc_class_inc=acc_class_inc)

    assert len(dataset) == len(scalar_meter)
    result_dict = scalar_meter.update_epoch_average_value()

    print(f"Task [{train_taskid + 1}/{GVM.cl_mngr.num_tasks}]:: Eval [{eval_taskid + 1:>{len(_tt := str(train_taskid + 1))}}/{_tt}]: eval_time={ttime() - _ttimer:.1f}s, {scalar_meter.format_outout(result_dict)}")

    result_dict['num_samples'] = len(dataset)

    return result_dict


def evaluate_tasks_sofar(GVM: GlobalVarsManager, train_taskid: int, model: VisionTransformer):
    print(f"*" * 90 + f" Start Evaluation " + "*" * 90)

    model = modify_head(GVM, model, training=False)
    # torch.cuda.empty_cache()
    average_acc_meter = misc.ScalarMeter(acc_task_inc="samp_avg:>6.2%", acc_class_inc="samp_avg:>6.2%")

    for eval_taskid in range(GVM.cl_mngr.current_taskid + 1):
        eval_task_classes = GVM.cl_mngr.get_classes(eval_taskid)
        one_result_dict = evaluate_one_task(GVM, train_taskid, eval_taskid, eval_task_classes, model)
        GVM.acc_mat_dict[f'AccTaskIncMat'][train_taskid, eval_taskid] = one_result_dict['acc_task_inc']
        GVM.acc_mat_dict[f'AccClassIncMat'][train_taskid, eval_taskid] = one_result_dict['acc_class_inc']
        average_acc_meter.add_step_value(**one_result_dict)
    model.remove_text_features()

    avg_result_dict = average_acc_meter.update_epoch_average_value()
    GVM.acc_mat_dict[f'AccTaskIncList'][train_taskid] = avg_result_dict['acc_task_inc']
    GVM.acc_mat_dict[f'AccClassIncList'][train_taskid] = avg_result_dict['acc_class_inc']


def task_ending_info(GVM: GlobalVarsManager):
    current_taskid = GVM.cl_mngr.current_taskid

    print(f"{'=' * 90} End of task [{current_taskid + 1}/{GVM.cl_mngr.num_tasks}] {'=' * 90}")
    print("".join([f"\n{_acc_name}(task {current_taskid + 1}):\n{_acc_mat}" for _acc_name, _acc_mat in GVM.acc_mat_dict.items()]))

    acc_info_dict = {
        'diag_task_avg_acc': float(np.diag(GVM.acc_mat_dict['AccTaskIncMat'])[:current_taskid + 1].mean()),
        'task_inc_last_acc': float(GVM.acc_mat_dict['AccTaskIncList'][current_taskid]),
        'task_inc_last_forg': misc.calc_forgetting(GVM.acc_mat_dict['AccTaskIncMat'], current_taskid),
        'class_inc_last_acc': float(GVM.acc_mat_dict['AccClassIncList'][current_taskid]),
        'class_inc_last_forg': misc.calc_forgetting(GVM.acc_mat_dict['AccClassIncMat'], current_taskid),
    }
    _formatter = misc.ScalarFormatter(sep=' | ', diag_task_avg_acc=">6.2%", task_inc_last_acc=">6.2%", class_inc_last_acc=">6.2%", task_inc_last_forg=">6.2%", class_inc_last_forg=">6.2%")

    print(f":: ** Results of task [{current_taskid + 1}]: [ {_formatter(**acc_info_dict)} ] **")
    print(f":: ** Time so far: {misc.format_duration(ttime() - GVM.cache_dict['exp_start_time'])} **")


def find_not_pretrained_params(model: VisionTransformer, pretrained: bool = True, pretrained_cfg: dict[str, str] = None, extra_pretrained_params: list[str] = []) -> list[str]:
    assert isinstance(extra_pretrained_params, (list, tuple))

    assert pretrained_cfg is not None

    if 'open_clip' in pretrained_cfg.get('hf_hub_filename', ''):
        _filename = timm.models._hub.HF_OPEN_CLIP_WEIGHTS_NAME
    else:
        _filename = timm.models._hub.HF_WEIGHTS_NAME
    pre_state_dict: OrderedDict[str, Tensor] = timm.models.load_state_dict_from_hf(pretrained_cfg['hf_hub_id'], _filename)

    if 'visual.class_embedding' in pre_state_dict.keys():
        pre_state_dict = timm.models.vision_transformer._convert_openai_clip(pre_state_dict, model)

    not_pretrained_params = []
    for n, p in model.named_parameters():
        if n not in pre_state_dict.keys() or not pretrained:
            not_pretrained_params.append(n)
        else:
            if p.shape != pre_state_dict[n].shape:
                not_pretrained_params.append(n)

    for n in deepcopy(not_pretrained_params):
        for _p in extra_pretrained_params:
            if _p in n:
                not_pretrained_params.remove(n)
    assert len(not_pretrained_params) > 0
    return not_pretrained_params


def get_param_id_dict(model: VisionTransformer, patterns: list[str]) -> dict[int, dict[Literal['name', 'shape'], str | list[int]]]:
    param_id_dict = {}
    for n, p in model.named_parameters():
        if p.requires_grad and any([_s in n for _s in patterns]):
            param_id_dict[id(p)] = {'name': n, 'shape': list(p.shape)}
    assert len(param_id_dict) > 0, f"{param_id_dict}"
    return param_id_dict


def get_head_dim_arg_dict(GVM: GlobalVarsManager, args: argparse.Namespace) -> dict[Literal['num_classes'], int]:
    head_dim_arg_dict = {}
    head_dim_type = args.head_dim_type

    match args.logit_type:
        case 'head_out':
            assert head_dim_type in ('task_classes', 'pretrained')

    match head_dim_type:
        case 'task_classes':
            head_dim_arg_dict['num_classes'] = len(current_task_classes) if args.seperate_head else len(GVM.cl_mngr.sofar_task_classes)
        case 'pretrained':
            pass
        case _:
            raise ValueError(head_dim_type)
    return head_dim_arg_dict


def modify_head(GVM: GlobalVarsManager, model: VisionTransformer, training: bool, ignore_requires_grad_check: bool = False, **kwargs):
    args: argparse.Namespace = GVM.args

    if training:
        _target_classes = kwargs['task_classes'] if args.seperate_head else GVM.cl_mngr.sofar_task_classes
    else:
        _target_classes = GVM.cl_mngr.sofar_task_classes

    if args.logit_type == 'head_out':
        if model.head.out_features != len(_target_classes):
            _mh = deepcopy(model.head)
            _mdevice = _mh.weight.device
            _mdtype = _mh.weight.dtype
            model.head = _mh.__class__(_mh.in_features, len(_target_classes), _mh.bias is not None, _mdevice, _mdtype)
            model.head.requires_grad_(_mh.weight.requires_grad)

            if training:
                if not ignore_requires_grad_check:
                    assert model.head.weight.requires_grad
            else:
                assert _mh.out_features == len(GVM.cl_mngr.current_task_classes), f"{_mh.out_features}, {len(GVM.cl_mngr.current_task_classes)}"
                _hw = torch.cat([GVM.param_dict[f'task_params_{_t}']['head.weight'].data.to(_mdevice, _mdtype) for _t in range(GVM.cl_mngr.current_taskid + 1)])
                assert model.head.weight.data.shape == _hw.shape
                model.head.weight.data = _hw

                if _mh.bias is not None:
                    _hb = torch.cat([GVM.param_dict[f'task_params_{_t}']['head.bias'].data.to(_mdevice, _mdtype) for _t in range(GVM.cl_mngr.current_taskid + 1)])
                    assert model.head.bias.data.shape == _hb.shape
                    model.head.bias.data = _hb
        else:
            pass
    else:
        raise ValueError(args.logit_type)

    return model


if __name__ == "__main__":
    args = get_args()
    seed_etc_options(args.seed)
    GVM = GlobalVarsManager()
    GVM.init_from_args(args)
    GVM.cache_dict['exp_start_time'] = ttime()

    if args.resume:
        raise NotImplementedError()

    for taskid, current_task_classes in GVM.cl_mngr:
        print(f"{'#' * 90} Task: [{taskid + 1}/{GVM.cl_mngr.num_tasks}] {'#' * 90}")
        print(f":: Current classes ({len(current_task_classes)}): {current_task_classes}")

        if not args.consecutive_training or taskid == 0:
            _prompt_args_dict = misc.get_specific_args_dict(args, 'prompt_')
            _other_args_dict = misc.get_specific_args_dict(args, 'logit_')
            _head_dim_arg_dict = get_head_dim_arg_dict(GVM, args)

            model: VisionTransformer = timm.create_model(args.model, pretrained=True, pretrained_strict=False, **_head_dim_arg_dict,
                                                         prompt_args_dict=_prompt_args_dict, other_args_dict=_other_args_dict)
            GVM.cache_dict['pretrained_cfg'] = deepcopy(model.pretrained_cfg)

        if args.use_compile:
            raise NotImplementedError()

        _not_pretrained_params = find_not_pretrained_params(model, pretrained_cfg=model.pretrained_cfg)
        GVM.cache_dict['not_pretrained_params'] = _not_pretrained_params

        if args.eval_only or (args.resume and f'task_params_{taskid}' in GVM.param_dict):
            GVM.update_label_maps(taskid, current_task_classes)
            # Load parameters for the model
            model.head.requires_grad_()
            model = modify_head(GVM, model, training=True, task_classes=current_task_classes)
            model.head.requires_grad_(False)
            _loaded_params = misc.load_params(GVM.param_dict, model, f'task_params_{taskid}')
            misc.check_param_loading(GVM.cache_dict['not_pretrained_params'], _loaded_params)
            if args.use_ncm:
                extract_class_features(GVM, model)
        else:
            # Training
            GVM.update_label_maps(taskid, current_task_classes)
            GVM.cache_dict['training_string'] = args.training_string
            misc.check_param_training(GVM.cache_dict['not_pretrained_params'], GVM.cache_dict['training_string'])
            if taskid > 0 and args.impt_enable:
                new_importance_dict = calc_importance_by_gradient(GVM, taskid, current_task_classes, model, args.impt_loss_new)
                GVM.cache_dict[f'new_importance_dict_{taskid}'] = new_importance_dict
                select_important_params(GVM, taskid, old_importance_dict, new_importance_dict, topk=args.impt_topk)

            model = train_one_task(GVM, taskid, current_task_classes, model)
            if taskid < GVM.cl_mngr.num_tasks - 1 and args.impt_enable:
                old_importance_dict = calc_importance_by_gradient(GVM, taskid, current_task_classes, model, args.impt_loss_old)
                GVM.cache_dict[f'old_importance_dict_{taskid}'] = old_importance_dict if taskid == 0 else accumulate_importance_dict(GVM, taskid, old_importance_dict)

        # Evaluate the model
        evaluate_tasks_sofar(GVM, taskid, model)
        task_ending_info(GVM)
