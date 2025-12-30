from collections import OrderedDict
from copy import deepcopy
from typing import Literal
from time import time as ttime

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import tqdm
from einops import rearrange, reduce, repeat

from .vit_builder import VisionTransformer
from .supc_loss import SupConLossByGPS
from . import misc


def calc_importance_by_gradient(GVM, taskid: int, task_classes: list[int], model: VisionTransformer, criterion_name: Literal['SupCon'], verbose: bool = True):
    from .dataset_utils import define_dataset
    
    args = GVM.args
    _tsfm = 'contrastive'
    _expansion = args.impt_supcon_repeat

    train_dataset = define_dataset(GVM, task_classes, training=True, transform_type=_tsfm, target_map_to_local=args.seperate_head, use_eval_transform=True, expand_times=_expansion, verbose=False)
    dataloader = DataLoader(train_dataset, batch_size=args.impt_batch_size, shuffle=args.impt_data_shuffle, num_workers=args.eval_workers, pin_memory=True, timeout=args.timeout if args.eval_workers > 0 else 0)

    is_trained_task = f'task_params_{taskid}' in GVM.param_dict

    scalar_meter = misc.ScalarMeter(loss="samp_avg:.4f", batch_time="step_sum:.3f", acc_top1="samp_avg:>6.2%")

    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train_eval import set_model_mode, modify_head
    
    prompt_params_str: list[str] = deepcopy(GVM.cache_dict['training_string'])
    prompt_params_str.remove('head')
    model: VisionTransformer = set_model_mode(GVM, model, training=True, training_string=prompt_params_str, ignore_pretrained_check=True)

    model = modify_head(GVM, model, training=True, task_classes=task_classes, ignore_requires_grad_check=True)

    if is_trained_task:
        head_params = {n.removeprefix('head.'): p for n, p in GVM.param_dict[f'task_params_{taskid}'].items() if 'head' in n}
        model.head.load_state_dict(head_params)

    criterion = SupConLossByGPS(normalize=True).cuda()

    param_groups = OrderedDict(filter(lambda n_p: n_p[1].requires_grad, model.named_parameters()))
    assert len(param_groups) > 0
    model.zero_grad(set_to_none=True)

    torch.cuda.empty_cache()
    _btimer = ttime()
    for i_batch, (images, target) in tqdm.tqdm(enumerate(dataloader, 1), total=len(dataloader), dynamic_ncols=True, disable=not GVM.args.show_bar):
        assert isinstance(images, (list, tuple)), type(images)
        assert len(images) == 2, len(images)
        images = torch.cat([images[0], images[1]], dim=0)

        images: Tensor = images.cuda(non_blocking=True)
        target: Tensor = target.cuda(non_blocking=True)

        feats: Tensor = model.encode_image(images, pre_logits=True)
        feats = rearrange(feats, '(b v) d -> b v d', b=target.shape[0], v=2)

        loss = criterion(feats, target)
        if torch.isnan(loss).item():
            raise ValueError("Loss is Nan")

        logits = torch.mean(model.head(feats), 1)
        assert logits.ndim == 2

        loss.backward()

        acc_top1, = misc.calc_accuracy(logits, target, topk=(1, ))
        batch_time = ttime() - _btimer

        scalar_meter.add_step_value(len(images), loss=loss.item(), batch_time=batch_time, acc_top1=acc_top1)
        _btimer = ttime()
    _epoch_scalar_str = scalar_meter.format_outout(scalar_meter.update_epoch_average_value())

    torch.cuda.empty_cache()

    importance_dict = OrderedDict()
    for n, p in param_groups.items():
        _pid = id(p)
        _grad = (p.grad ** 2).sqrt()
        if args.impt_norm_layer_grad:
            _grad = (_grad - _grad.amin()) / (_grad.amax() - _grad.amin())

        importance_dict[_pid] = _grad.cpu()

    return importance_dict


def accumulate_importance_dict(GVM, taskid: int, curr_importance_dict: OrderedDict[int, Tensor]):
    impt_momentum_old: float = GVM.args.impt_momentum_old
    assert (_i1 := f'old_importance_dict_{taskid}' not in GVM.cache_dict) and (_i2 := f'old_importance_dict_{taskid - 1}' in GVM.cache_dict), f"old_importance_dict_{taskid} in GVM: {_i1}; old_importance_dict_{taskid - 1} in GVM: {_i2}"
    fused_importance_dict = OrderedDict()
    for _pid in curr_importance_dict.keys():
        previous_val = GVM.cache_dict[f'old_importance_dict_{taskid - 1}'][_pid]
        last_val = curr_importance_dict[_pid]
        if impt_momentum_old == 1:
            fused_importance_dict[_pid] = last_val
        elif impt_momentum_old == 0:
            fused_importance_dict[_pid] = previous_val
        else:
            assert 0 < impt_momentum_old < 1
            fused_importance_dict[_pid] = (1 - impt_momentum_old) * previous_val + impt_momentum_old * last_val
    assert list(fused_importance_dict.keys()) == list(curr_importance_dict.keys())
    return fused_importance_dict


def select_important_params(GVM, taskid: int, importance_dict_oldtask: OrderedDict[str | int, Tensor], importance_dict_newtask: OrderedDict[str, Tensor], topk: float, verbose: bool = True):
    def get_high_importance_mask_by_elem(stack_importance: Tensor, num_top_elems: int, topk_ratio: float = None):
        # reshaped_importance: [L, M, D]
        L, M, D = stack_importance.shape
        assert L == GVM.args.prompt_end_block - GVM.args.prompt_start_block + 1

        ind_topk = rearrange(stack_importance, 'L M D -> (L M D)').topk(num_top_elems).indices
        mask = torch.zeros(stack_importance.numel(), dtype=torch.bool, device=stack_importance.device)
        mask[ind_topk] = True
        mask = rearrange(mask, '(L M D) -> L M D', L=L, M=M, D=D)
        assert torch.count_nonzero(mask) == num_top_elems
        return mask

    def get_high_importance_mask_by_token(stack_importance: Tensor, num_top_elems: int, select_level: str):
        # reshaped_importance: [L, M, D]
        L, M, D = stack_importance.shape
        assert L == GVM.args.prompt_end_block - GVM.args.prompt_start_block + 1

        if select_level =='token':
            ind_topk = reduce(stack_importance, 'L M D -> (L M)', reduction='mean').topk(num_top_elems).indices
            mask = torch.zeros(L * M, dtype=torch.bool, device=stack_importance.device)
            mask[ind_topk] = True
            mask = repeat(mask, '(L M) -> L M D', L=L, M=M, D=D)
            assert torch.count_nonzero(mask) == num_top_elems * D
        elif select_level == 'dime':
            ind_topk = reduce(stack_importance, 'L M D -> (L D)', reduction='mean').topk(num_top_elems).indices
            mask = torch.zeros(L * D, dtype=torch.bool, device=stack_importance.device)
            mask[ind_topk] = True
            mask = repeat(mask, '(L D) -> L M D', L=L, M=M, D=D)
            assert torch.count_nonzero(mask) == num_top_elems * M
        else:
            raise ValueError(select_level)
        return mask

    args = GVM.args
    stack_importance_old = torch.cat(list(importance_dict_oldtask.values()))  # [L, M, D]
    stack_importance_new = torch.cat(list(importance_dict_newtask.values()))  # [L, M, D]

    if args.impt_select_level == 'elem':
        assert 0 < topk < 1
        num_top_elems = int(stack_importance_old.numel() * topk)
        assert num_top_elems > 0, num_top_elems
        mask_old = get_high_importance_mask_by_elem(stack_importance_old, num_top_elems)  # [L, M, D]
        mask_new = get_high_importance_mask_by_elem(stack_importance_new, num_top_elems)  # [L, M, D]
    elif args.impt_select_level in ('token', 'dime'):
        assert topk == int(topk) and topk >= 1
        num_top_elems = int(topk)
        mask_old = get_high_importance_mask_by_token(stack_importance_old, num_top_elems, args.impt_select_level)  # [L, M, D]
        mask_new = get_high_importance_mask_by_token(stack_importance_new, num_top_elems, args.impt_select_level)  # [L, M, D]
    else:
        raise NotImplementedError()

    mask_intersection = torch.bitwise_and(mask_old, mask_new)
    mask_nonunionset = ~torch.bitwise_or(mask_old, mask_new)
    mask_only_old = torch.bitwise_and(mask_old, ~mask_intersection)
    mask_only_new = torch.bitwise_and(mask_new, ~mask_intersection)

    GVM.cache_dict['mask_dict'] = OrderedDict()
    for _pid, _mi, _mu, _mo, _mn in zip(importance_dict_oldtask.keys(), mask_intersection, mask_nonunionset, mask_only_old, mask_only_new):
        GVM.cache_dict['mask_dict'][_pid] = torch.stack([_mi, _mu, _mo, _mn]).cuda()  # [4, M, D]
    assert len(GVM.cache_dict['mask_dict']) == GVM.args.prompt_end_block - GVM.args.prompt_start_block + 1

    return GVM.cache_dict['mask_dict']
