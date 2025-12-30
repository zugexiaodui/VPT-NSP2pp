import torchvision.transforms as T
from copy import deepcopy
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from .dataset_builder import ClassIncremantalDataset
from .supc_loss import TwoCropTransform


def define_dataset(GVM, task_classes: list[int], training: bool, transform_type: str = 'timm', target_map_to_local: bool = True,
                   use_eval_transform: bool = False, expand_times: int = 1, verbose: bool = False, **kwargs) -> ClassIncremantalDataset:
    _current_dataset = GVM.args.dataset
    match GVM.args.interp_mode:
        case 'bilinear':
            interp_mode = T.InterpolationMode.BILINEAR
        case 'bicubic':
            interp_mode = T.InterpolationMode.BICUBIC
        case _:
            raise ValueError(GVM.args.interp_mode)
    bilinear = T.InterpolationMode.BILINEAR

    match transform_type:
        case 'timm':
            transforms = create_transform(**resolve_data_config(GVM.cache_dict['pretrained_cfg']), is_training=training if not use_eval_transform else False)
        case 'autoaug':
            dmean: tuple[float] = GVM.cache_dict['pretrained_cfg']['mean']
            dstd: tuple[float] = GVM.cache_dict['pretrained_cfg']['std']
            if training and not use_eval_transform:
                match _current_dataset:
                    case 'cifar100' | 'eurosat' | 'resisc':
                        transforms = T.Compose([T.AutoAugment(T.AutoAugmentPolicy.CIFAR10, bilinear), T.RandomResizedCrop((224, 224), interpolation=interp_mode, antialias=True), T.ToTensor(), T.Normalize(dmean, dstd)])
                    case 'imagenet_r' | 'sdomainet' | 'omni':
                        transforms = T.Compose([T.AutoAugment(T.AutoAugmentPolicy.IMAGENET, bilinear), T.RandomResizedCrop((224, 224), interpolation=interp_mode, antialias=True), T.ToTensor(), T.Normalize(dmean, dstd)])
                    case _:
                        raise NotImplementedError(_current_dataset)
            else:
                match _current_dataset:
                    case 'cifar100' | 'eurosat' | 'resisc':
                        transforms = T.Compose([T.Resize((224, 224), antialias=True, interpolation=interp_mode), T.ToTensor(), T.Normalize(dmean, dstd)])
                    case 'imagenet_r' | 'sdomainet' | 'omni':
                        transforms = T.Compose([T.Resize((256, 256), antialias=True, interpolation=interp_mode), T.CenterCrop(224), T.ToTensor(), T.Normalize(dmean, dstd)])
                    case _:
                        raise NotImplementedError(_current_dataset)
        case 'prototype':
            assert not training or use_eval_transform, "Only used for extracting prototypes"
            match _current_dataset:
                case 'cifar100' | 'eurosat' | 'resisc':
                    transforms = T.Compose([T.ToTensor(), T.Resize((224, 224), antialias=True)])
                case 'imagenet_r' | 'sdomainet' | 'omni':
                    transforms = T.Compose([T.Resize((256, 256), antialias=True), T.CenterCrop((224, 224)), T.ToTensor()])
                case _:
                    raise NotImplementedError(_current_dataset)
        case 'contrastive':
            dmean: tuple[float] = GVM.cache_dict['pretrained_cfg']['mean']
            dstd: tuple[float] = GVM.cache_dict['pretrained_cfg']['std']
            match _current_dataset:
                case 'cifar100' | 'eurosat' | 'resisc':
                    if GVM.args.impt_contrast_augment == 'autoaug':
                        transforms = TwoCropTransform(T.Compose([T.AutoAugment(T.AutoAugmentPolicy.CIFAR10, bilinear), T.Resize((224, 224), antialias=True, interpolation=interp_mode), T.ToTensor(), T.Normalize(dmean, dstd)]))
                    elif GVM.args.impt_contrast_augment == 'original':
                        transforms = TwoCropTransform(T.Compose([T.RandomResizedCrop(size=224, scale=(0.2, 1.)), T.RandomHorizontalFlip(),
                                                                 T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                                                 T.RandomGrayscale(p=0.2), T.ToTensor(), T.Normalize(dmean, dstd)]))
                    else:
                        raise ValueError(GVM.args.impt_contrast_augment)
                case 'imagenet_r' | 'sdomainet' | 'omni':
                    if GVM.args.impt_contrast_augment == 'autoaug':
                        transforms = TwoCropTransform(T.Compose([T.AutoAugment(T.AutoAugmentPolicy.IMAGENET, bilinear), T.Resize((256, 256), antialias=True, interpolation=interp_mode), T.CenterCrop(224), T.ToTensor(), T.Normalize(dmean, dstd)]))
                    elif GVM.args.impt_contrast_augment == 'original':
                        transforms = TwoCropTransform(T.Compose([T.RandomResizedCrop(size=224, scale=(0.2, 1.)), T.RandomHorizontalFlip(),
                                                                 T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                                                 T.RandomGrayscale(p=0.2), T.ToTensor(), T.Normalize(dmean, dstd)]))
                    else:
                        raise ValueError(GVM.args.impt_contrast_augment)
                case _:
                    raise NotImplementedError(_current_dataset)
        case _:
            raise NotImplementedError(f"{transform_type}")

    class TargetTransform():
        def __init__(self, label_map_g2l: dict[int, tuple[int, int, int]], target_map_to_local: bool) -> None:
            self.label_map_g2l = deepcopy(label_map_g2l)  # {original_label: (taskid, local_label, global_label)}
            self.target_map_to_local = target_map_to_local

        def __call__(self, target: int):
            if self.target_map_to_local:
                return self.label_map_g2l[target][1]
            else:
                return self.label_map_g2l[target][2]

        def __repr__(self) -> str:
            label_map = {k: v[1] if self.target_map_to_local else v[2] for k, v in self.label_map_g2l.items()}
            _repr = str(label_map)
            return _repr

    target_transforms = TargetTransform(GVM.label_map_g2l, target_map_to_local)

    _mode = 'train' if training else 'eval'
    dataset = ClassIncremantalDataset(GVM.path_data_dict[_mode], task_classes, transforms, target_transforms, expand_times=expand_times, verbose=verbose, return_index=False, sample_type=GVM.args.sample_type)

    return dataset
