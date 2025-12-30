from collections import OrderedDict
from typing import Literal

import numpy as np
import scipy.ndimage
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from einops import rearrange

from .vit_builder import VisionTransformer
from . import vit_builder


def get_interm_tensor_dict(GVM, model: VisionTransformer, null_param_id_dict: dict, verbose: bool = True) -> dict[int, Tensor]:
    from .dataset_utils import define_dataset
    
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train_eval import set_model_mode
    
    interm_tensor_dict: dict[int, dict[str, Tensor]] = {}

    _pre_tokens = model.num_pre_tokens
    edim = model.num_features
    nheads = model.num_heads
        
    def _forward_hook(module: nn.Module, args: tuple[Tensor], output: Tensor):

        if isinstance(module, nn.Linear):
            pass
        elif isinstance(module, vit_builder.IntermReader):
            _pid = module.dst_param_id
            _mname = module.module_name
            _interm_tensor: Tensor = args[0]  # [B, H, N+M, d]
            if _mname == 'interm_reader_1' and GVM.args.prompt_len > 0:
                w_qkv = module.other_args['w_qkv'].detach()
                w_k: Tensor = rearrange(w_qkv, '(n do) di -> n do di', n=3, do=edim, di=edim).unbind(0)[1]  # [D, D]
                w_k = rearrange(w_k, '(h d) D -> h d D', h=nheads, d=edim//nheads, D=edim)  # [H, D, d]
                w_k = rearrange(w_k, 'h d D -> b h d D', b=_interm_tensor.shape[0])  # [B, H, d, D]

                _interm_tensor = _interm_tensor[:, :, :_pre_tokens]  # [B, H, N, d]
                assert _interm_tensor.shape[2] == _pre_tokens, f"{_interm_tensor.shape}"
                _interm_tensor = _interm_tensor @ w_k  # [B, H, N, d] -> [B, H, N, D]
                _interm_tensor = rearrange(_interm_tensor, 'b h n d -> (b h n) d')
                _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]

            if _mname == 'interm_reader_2' and GVM.args.prompt_len > 0:
                _interm_tensor = _interm_tensor[:, :, :_pre_tokens, _pre_tokens:]  # [B, H, N, M]
                assert _interm_tensor.shape[-1] == GVM.args.prompt_len
                _interm_tensor = rearrange(_interm_tensor, 'b h n m -> (b h n) m')
                _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]

            if _mname in ('interm_reader_1', 'interm_reader_2') and GVM.args.adap_alpha > 0:
                # _interm_tensor: [B, N, D|reduced_D]
                assert _interm_tensor.ndim == 3 and _interm_tensor.shape[1] == _pre_tokens, _interm_tensor.shape
                _interm_tensor = rearrange(_interm_tensor, 'b n d -> (b n) d')
                _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]

            assert _mname in ('interm_reader_1', 'interm_reader_2')
            if _pid not in interm_tensor_dict:
                interm_tensor_dict[_pid] = {}
            if _mname not in interm_tensor_dict[_pid]:
                interm_tensor_dict[_pid][_mname] = torch.zeros_like(_interm_tensor)
            interm_tensor_dict[_pid][_mname] += _interm_tensor
        else:
            raise NotImplementedError()

    _handle_list: list[torch.utils.hooks.RemovableHandle] = []
    for n, m in model.named_modules():
        if 'interm_reader' in n and isinstance(m, vit_builder.IntermReader):
            _handle_list.append(m.register_forward_hook(_forward_hook))

    model = set_model_mode(GVM, model, training=False)

    args = GVM.args
    train_dataset = define_dataset(GVM, GVM.cl_mngr.current_task_classes, training=True, transform_type=args.transform_type, target_map_to_local=args.seperate_head, use_eval_transform=True, expand_times=1, verbose=False)
    dataset = train_dataset

    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=args.null_data_shuffle, num_workers=args.eval_workers, pin_memory=True, timeout=args.timeout if args.eval_workers > 0 else 0)

    for nb, (img, _) in enumerate(dataloader, 1):
        with torch.no_grad():
            img: Tensor
            model(img.cuda(non_blocking=True))
        if GVM.args.null_interm_batches >= 1:
            if nb >= GVM.args.null_interm_batches:
                break

    for _h in _handle_list:
        _h.remove()

    assert len(interm_tensor_dict) > 0
    assert list(interm_tensor_dict.keys()) == list(null_param_id_dict.keys()), f"{interm_tensor_dict.keys()}; {null_param_id_dict}"

    if (_k := 'interm_sample_list') not in GVM.cache_dict:
        GVM.cache_dict[_k] = []
    GVM.cache_dict[_k].append(len(dataloader.dataset))

    return interm_tensor_dict


def accumulate_interm_tensor_dict(GVM, cached_interm_tensor_dict: dict[int, dict[str, Tensor]], new_interm_tensor_dict: dict[int, dict[str, Tensor]], verbose: bool = True) -> dict[int, dict[str, Tensor]]:
    assert len(new_interm_tensor_dict) > 0
    args = GVM.args

    if cached_interm_tensor_dict == {}:
        merged_interm_tensor_dict = new_interm_tensor_dict
    else:
        assert (_lc := list(cached_interm_tensor_dict.keys())) == (_ln := list(new_interm_tensor_dict.keys())), f"{_lc}, {_ln}"
        merged_interm_tensor_dict: dict[int, Tensor] = {}
        for _pid in cached_interm_tensor_dict.keys():
            merged_interm_tensor_dict[_pid] = {}
            for _mname in cached_interm_tensor_dict[_pid].keys():
                _cached_tensor = cached_interm_tensor_dict[_pid][_mname]
                _new_tensor = new_interm_tensor_dict[_pid][_mname]
                assert _cached_tensor.shape == _new_tensor.shape

                match args.null_interm_accum:
                    case 'sum':
                        merged_interm_tensor_dict[_pid][_mname] = _cached_tensor + _new_tensor
                    case 'mean':
                        _num_list: list[int] = GVM.cache_dict['interm_sample_list']
                        merged_interm_tensor_dict[_pid][_mname] = sum(_num_list[:-1]) / sum(_num_list) * _cached_tensor + _num_list[-1] / sum(_num_list) * _new_tensor
                    case 'onlynew':
                        merged_interm_tensor_dict[_pid][_mname] = _new_tensor
                    case 'onlyold':
                        merged_interm_tensor_dict[_pid][_mname] = _cached_tensor
                    case _:
                        raise ValueError()

    return merged_interm_tensor_dict


def get_update_projection_dict(GVM, null_param_id_dict: dict, interm_tensor_dict: dict[int, dict[str, Tensor]], verbose: bool = True) -> dict[int, dict[str, Tensor]]:
    args = GVM.args
    update_proj_dict = {}

    def adaptive_threshold(svals: torch.Tensor, offset: float = 0):
        points: np.ndarray = svals.cpu().numpy()
        assert points.ndim == 1
        if len(points) >= 128:
            fil_points = scipy.ndimage.gaussian_filter1d(points, sigma=10)
            _delta = 1
            diff_o1 = fil_points[:-_delta] - fil_points[_delta:]
            diff_o2 = diff_o1[:-1] - diff_o1[1:]
            _drop_ratio = 0.03
            drop_num = int(len(points) * _drop_ratio / 2)
            assert len(points) - drop_num >= 10
            valid_o2 = diff_o2[drop_num:-drop_num]
            thres_val = points[np.argmax(valid_o2) + int((len(points) - len(valid_o2)) / 2)]
        else:
            diff_o1 = points[:-1] - points[1:]
            diff_o2 = diff_o1[:-1] - diff_o1[1:]
            thres_val = points[np.argmax(diff_o2) + int((len(points) - len(diff_o2)) / 2)]
        i_thres = np.arange(len(points))[points >= thres_val].max()
        if 0 <= offset < 1:
            i_thres = min(i_thres + int(offset * (len(points) - i_thres)), len(points) - 1)
        else:
            i_thres = max(min(i_thres + int(offset), len(points) - 1), 0)

        zero_idx = np.zeros(len(points), dtype=np.int64)
        zero_idx[i_thres:] = 1
        zero_idx = torch.as_tensor(torch.from_numpy(zero_idx), dtype=torch.bool, device=svals.device)
        return zero_idx

    svals_dict = OrderedDict()
    zero_idx_dict = OrderedDict()
    for _pid in interm_tensor_dict.keys():
        update_proj_dict[_pid] = {}
        for _mname in interm_tensor_dict[_pid].keys():
            U, S, Vt = torch.linalg.svd(interm_tensor_dict[_pid][_mname], full_matrices=True)  # A=U@diag(S)@Vt
            S: Tensor
            Vt: Tensor

            thres_value = {'interm_reader_1': args.null_thres_value1, 'interm_reader_2': args.null_thres_value2}[_mname]
            match args.null_thres_mode:
                case 'times':
                    zero_idx = S <= S[-1] * int(thres_value)
                case 'num':
                    zero_idx = S <= S[-int(thres_value)]
                case 'pct':
                    zero_idx = S <= S[-round(thres_value / 100. * S.shape[0])]
                case 'val':
                    zero_idx = S <= thres_value
                case 'adaptive':
                    zero_idx = adaptive_threshold(S, offset=thres_value)
                case _:
                    raise ValueError(args.null_thres_mode)
            zero_idx: torch.BoolTensor
            assert torch.count_nonzero(zero_idx) > 0, f"{zero_idx}, {type(zero_idx)}, {torch.count_nonzero(zero_idx)}"

            svals_dict[_dkey := f"{null_param_id_dict[_pid]['name']}-{_mname}"] = S.cpu().clone()
            zero_idx_dict[_dkey] = zero_idx.cpu().clone()

            basis = Vt[zero_idx]
            proj = basis.T @ basis
            proj = proj / torch.norm(proj)

            assert proj.shape[0] == proj.shape[1], proj.shape

            if args.impt_enable:
                update_proj_dict[_pid][_mname] = proj.detach()
            else:
                null_alpha: float = {'interm_reader_1': args.null_alpha1, 'interm_reader_2': args.null_alpha2}[_mname]
                update_proj_dict[_pid][_mname] = null_alpha * proj.detach() + (1 - null_alpha) * torch.eye(proj.shape[0], device=proj.device, dtype=proj.dtype)

    return update_proj_dict
