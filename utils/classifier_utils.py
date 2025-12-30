import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from .vit_builder import VisionTransformer
from . import misc


def ncm_classifier(GVM, test_feats: Tensor) -> Tensor:
    assert test_feats.ndim == 2 and test_feats.shape[1] in (768, 512)  # [B, D]
    _train_proto: Tensor = GVM.cache_dict['prototypes']['proto'].cuda()  # [C, D]
    _train_class: Tensor = GVM.cache_dict['prototypes']['class'].cuda()  # [C]
    cos_sim: Tensor = torch.vmap(lambda _f: F.cosine_similarity(_f, _train_proto))(test_feats.unsqueeze(1))  # [B, C]
    assert cos_sim.ndim == 2 and cos_sim.shape[0] == test_feats.shape[0] and cos_sim.shape[1] == _train_proto.shape[0], f"{cos_sim.shape}"
    logits = cos_sim[:, _train_class]
    logits = logits * torch.tensor(GVM.args.logit_scale, dtype=logits.dtype, device=logits.device).exp()
    return logits


def extract_class_features(GVM, model: VisionTransformer, verbose: bool = True) -> None:
    from .dataset_utils import define_dataset
    
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train_eval import set_model_mode
    
    model = set_model_mode(GVM, model, training=False)

    dataset = define_dataset(GVM, GVM.cl_mngr.current_task_classes, training=True, transform_type=GVM.args.transform_type, target_map_to_local=False, use_eval_transform=True, expand_times=1)
    dataloader = DataLoader(dataset, batch_size=GVM.args.eval_batch_size, shuffle=False, num_workers=GVM.args.eval_workers, pin_memory=True, timeout=GVM.args.timeout if GVM.args.eval_workers > 0 else 0)

    feats = torch.empty([len(dataset), model.embed_dim], dtype=torch.float32)
    label = torch.empty([len(dataset)], dtype=torch.long)

    smp_idx = 0
    for img, lbl in dataloader:
        with torch.no_grad():
            img: Tensor
            lbl: Tensor
            _feat = model.encode_image(img.cuda(non_blocking=True), pre_logits=True).cpu()
            for _f, _l in zip(_feat, lbl):
                feats[smp_idx] = _f
                label[smp_idx] = _l
                smp_idx += 1
    assert smp_idx == len(dataset)

    if GVM.args.refine_head:
        _mean_list = []
        _cov_list = []
        _class_list = []
        for _l in label.unique():
            _cls_feats = feats[label == _l]
            _mean_list.append(torch.mean(_cls_feats, dim=0, keepdim=False))
            _cov_list.append(torch.cov(torch.tensor(_cls_feats, dtype=torch.float64).T) + torch.eye(_cls_feats.shape[-1]) * 1e-4)
            _class_list.append(_l)
        _mean_list = torch.stack(_mean_list)
        _cov_list = torch.stack(_cov_list)
        _class_list = torch.stack(_class_list)

        _key = 'class_features'
        if _key not in GVM.cache_dict:
            GVM.cache_dict[_key] = {'mean': _mean_list, 'cov': _cov_list, 'class': _class_list}
        else:
            GVM.cache_dict[_key]['mean'] = torch.cat([GVM.cache_dict[_key]['mean'], _mean_list])
            GVM.cache_dict[_key]['cov'] = torch.cat([GVM.cache_dict[_key]['cov'], _cov_list])
            GVM.cache_dict[_key]['class'] = torch.cat([GVM.cache_dict[_key]['class'], _class_list])
            assert len(GVM.cache_dict[_key]['mean']) == len(GVM.cache_dict[_key]['cov']) == len(GVM.cache_dict[_key]['class']) == len(GVM.cl_mngr.sofar_task_classes), f"{len(GVM.cache_dict[_key]['mean'])}, {len(GVM.cache_dict[_key]['cov'])}, {len(GVM.cache_dict[_key]['class'])}, {len(GVM.cl_mngr.sofar_task_classes)}"

    if GVM.args.use_ncm:
        _proto_list = []
        _class_list = []
        for _l in label.unique():
            _proto_list.append(torch.mean(feats[label == _l], dim=0, keepdim=True))  # [1, 768]
            _class_list.append(_l)
        _proto_list = torch.cat(_proto_list)
        _class_list = torch.stack(_class_list)

        _key = 'prototypes'
        if _key not in GVM.cache_dict:
            GVM.cache_dict[_key] = {'proto': _proto_list, 'class': _class_list}
        else:
            GVM.cache_dict[_key]['proto'] = torch.cat([GVM.cache_dict[_key]['proto'], _proto_list])
            GVM.cache_dict[_key]['class'] = torch.cat([GVM.cache_dict[_key]['class'], _class_list])
        assert len(GVM.cache_dict[_key]['proto']) == len(GVM.cache_dict[_key]['class']) == len(GVM.cl_mngr.sofar_task_classes)

    return None


def refine_head(GVM, model: VisionTransformer):
    from torch.distributions.multivariate_normal import MultivariateNormal
    
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train_eval import modify_head
    
    feats_mean: Tensor = GVM.cache_dict['class_features']['mean']
    feats_cov: Tensor = GVM.cache_dict['class_features']['cov']
    feats_class: Tensor = GVM.cache_dict['class_features']['class']
    assert len(feats_class.unique()) == len(GVM.cl_mngr.sofar_task_classes)

    stat_dataset = TensorDataset(feats_mean, feats_cov, feats_class)

    model = modify_head(GVM, model, training=False)
    mhead = model.head

    mhead.train()
    mhead.cuda()
    mhead.requires_grad_()

    optimizer = create_optimizer_v2(mhead, opt='sgd', lr=0.001, weight_decay=1e-4, momentum=0.9)
    scheduler, num_epochs = create_scheduler_v2(optimizer, 'multistep', num_epochs=50, decay_milestones=[999,], decay_rate=0.1)
    criterion = nn.CrossEntropyLoss().cuda()

    scalar_meter = misc.ScalarMeter(loss="samp_avg:.4f", acc_top1="samp_avg:>6.2%")
    for epoch in range(1, num_epochs + 1):
        scheduler.step(epoch)

        smp_inp = []
        smp_tgt = []
        assert len(stat_dataset) == len(GVM.cl_mngr.sofar_task_classes)
        _ns = 256
        for _cmean, _ccov, _cclass in stat_dataset:
            m = MultivariateNormal(_cmean.float(), _ccov.float())
            _smp = m.sample(sample_shape=(_ns,))
            smp_inp.append(_smp)
            smp_tgt.append(torch.as_tensor([_cclass,] * _ns, dtype=torch.long))
        smp_inp = torch.cat(smp_inp)
        smp_tgt = torch.cat(smp_tgt)

        train_data = TensorDataset(smp_inp, smp_tgt)
        assert len(train_data) == len(stat_dataset) * _ns
        dataloader = DataLoader(train_data, batch_size=256, shuffle=True)

        for inp, tgt in dataloader:
            out: Tensor = mhead(inp.cuda(non_blocking=True))
            if model.logit_type == 'head_out':
                logits = out
            elif model.logit_type == 'sim_imgtext':
                logits = model.forward_logits(out)
            loss: Tensor = criterion(logits / GVM.args.temperature, tgt.cuda(non_blocking=True))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_top1, = misc.calc_accuracy(logits.cpu(), tgt.cpu(), topk=(1, ))
            scalar_meter.add_step_value(len(inp), loss=loss.item(), acc_top1=acc_top1)
            # break # DEBUG
        if (epoch % 10 == 0 or epoch == num_epochs):
            print(f":: refine_head: epoch [{epoch}/{num_epochs}]: {scalar_meter.format_outout(scalar_meter.update_epoch_average_value())}")
