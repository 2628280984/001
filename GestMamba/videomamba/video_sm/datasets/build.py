import os
from torchvision import transforms
from .transforms import *
from .masking_generator import (
    TubeMaskingGenerator, RandomMaskingGenerator,
    TubeRowMaskingGenerator,
    RandomRowMaskingGenerator
)
from .mae import VideoMAE

from .general import SMVideoClsDataset, SMRawFrameClsDataset
from .ego import EgoVideoClsDataset, EgoRawFrameClsDataset
from .general_muti import MMRawFrameClsDataset, MMVideoClsDataset
from .ego_muti import mutiEgoVideoClsDataset, mutiEgoRawFrameClsDataset



class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        if args.color_jitter > 0:
            self.transform = transforms.Compose([                            
                self.train_augmentation,
                GroupColorJitter(args.color_jitter),
                GroupRandomHorizontalFlip(flip=args.flip),
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([                            
                self.train_augmentation,
                GroupRandomHorizontalFlip(flip=args.flip),
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                normalize,
            ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        elif args.mask_type == 'random':
            self.masked_position_generator = RandomMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        elif args.mask_type == 'tube_row':
            self.masked_position_generator = TubeRowMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        elif args.mask_type == 'random_row':
            self.masked_position_generator = RandomRowMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        elif args.mask_type in 'attention':
            self.masked_position_generator = None

    def __call__(self, images):
        process_data, _ = self.transform(images)
        if self.masked_position_generator is None:
            return process_data, -1
        else:
            return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        prefix=args.prefix,
        split=args.split,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        num_segments=args.num_segments,
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=args.use_decord,
        lazy_init=False,
        num_sample=args.num_sample)
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args):
    print(f'Use Dataset: {args.data_set}')
    if args.data_set in ['NV', 'ISO', 'THU', 'FPHA'] and args.fusion is False:
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.txt')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.txt')
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'valid.txt')

        if args.use_decord:
            func = SMVideoClsDataset
        else:
            func = SMRawFrameClsDataset

        dataset = func(
            anno_path=anno_path,
            prefix=args.prefix,
            split=args.split,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            filename_tmpl=args.filename_tmpl,
            dataset=args.data_set,
            args=args)
        nb_classes = args.nb_classes

    elif args.data_set == 'Ego' and args.fusion is False:
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.txt')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.txt')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'valid.txt')

        if args.use_decord:
            func = EgoVideoClsDataset
        else:
            func = EgoRawFrameClsDataset

        dataset = func(
            anno_path=anno_path,
            prefix=args.prefix,
            split=args.split,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            filename_tmpl=args.filename_tmpl,
            dataset=args.data_set,
            args=args)
        nb_classes = 83

    elif args.data_set in ['NV', 'ISO', 'THU', 'FPHA'] and args.fusion is True:
        mode = None
        anno_path = None
        depth_anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.txt')
            depth_anno_path = os.path.join(args.depth_data_path, 'train.txt')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.txt')
            depth_anno_path = os.path.join(args.depth_data_path, 'test.txt')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'valid.txt')
            depth_anno_path = os.path.join(args.depth_data_path, 'valid.txt')

        if args.use_decord:
            func = MMVideoClsDataset
        else:
            func = MMRawFrameClsDataset

        dataset = func(
            anno_path=anno_path,
            depth_anno_path=depth_anno_path,
            prefix=args.prefix,
            depth_prefix=args.depth_prefix,
            split=args.split,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            filename_tmpl=args.filename_tmpl,
            depth_filename_tmpl=args.depth_filename_tmpl,
            dataset=args.data_set,
            args=args
            )
        nb_classes = args.nb_classes
    elif args.data_set == 'Ego' and args.fusion is True:
        mode = None
        anno_path = None
        depth_anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.txt')
            depth_anno_path = os.path.join(args.depth_data_path, 'train.txt')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.txt')
            depth_anno_path = os.path.join(args.depth_data_path, 'test.txt')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'valid.txt')
            depth_anno_path = os.path.join(args.depth_data_path, 'valid.txt')

        if args.use_decord:
            func = mutiEgoVideoClsDataset
        else:
            func = mutiEgoRawFrameClsDataset

        dataset = func(
            anno_path=anno_path,
            depth_anno_path=depth_anno_path,
            prefix=args.prefix,
            depth_prefix=args.depth_prefix,
            split=args.split,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            filename_tmpl=args.filename_tmpl,
            depth_filename_tmpl=args.depth_filename_tmpl,
            dataset=args.data_set,
            args=args
            )
        nb_classes = 83

    else:
        print(f'Wrong: {args.data_set}')
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
