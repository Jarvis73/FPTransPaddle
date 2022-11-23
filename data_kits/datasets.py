from paddle.io import DataLoader

from constants import data_dir, lists_dir
from data_kits import transformation as tf
from data_kits import voc_coco as pfe

DATA_DIR = {
    "PASCAL": data_dir / "VOCdevkit/VOC2012",
    "COCO": data_dir / "COCO",
}
DATA_LIST = {
    "PASCAL": {
        "train": lists_dir / "pascal/voc_sbd_merge_noduplicate.txt",
        "test": lists_dir / "pascal/val.txt",
        "eval_online": lists_dir / "pascal/val.txt"
    },
    "COCO": {
        "train": lists_dir / "coco/train_data_list.txt",
        "test": lists_dir / "coco/val_data_list.txt",
        "eval_online": lists_dir / "coco/val_data_list.txt"
    },
}
MEAN = [0.485, 0.456, 0.406]    # list, normalization mean in data preprocessing
STD = [0.229, 0.224, 0.225]     # list, normalization std in data preprocessing


def get_train_transforms(opt, height, width):
    supp_transform = tf.Compose([tf.RandomResize(opt.scale_min, opt.scale_max),
                                 tf.RandomRotate(opt.rotate, pad_type=opt.pad_type),
                                 tf.RandomGaussianBlur(),
                                 tf.RandomHorizontallyFlip(),
                                 tf.RandomCrop(height, width, check=True, center=True, pad_type=opt.pad_type),
                                 tf.ToTensor(mask_dtype='float'),   # support mask using float
                                 tf.Normalize(MEAN, STD)], processer=opt.proc)

    query_transform = tf.Compose([tf.RandomResize(opt.scale_min, opt.scale_max),
                                  tf.RandomRotate(opt.rotate, pad_type=opt.pad_type),
                                  tf.RandomGaussianBlur(),
                                  tf.RandomHorizontallyFlip(),
                                  tf.RandomCrop(height, width, check=True, center=True, pad_type=opt.pad_type),
                                  tf.ToTensor(mask_dtype='long'),   # query mask using long
                                  tf.Normalize(MEAN, STD)], processer=opt.proc)

    return supp_transform, query_transform


def get_val_transforms(opt, height, width):
    supp_transform = tf.Compose([tf.Resize(height, width),
                                 tf.ToTensor(mask_dtype='float'),   # support mask using float
                                 tf.Normalize(MEAN, STD)], processer=opt.proc)

    query_transform = tf.Compose([tf.Resize(height, width, do_mask=False),  # keep mask the original size
                                  tf.ToTensor(mask_dtype='long'),   # query mask using long
                                  tf.Normalize(MEAN, STD)], processer=opt.proc)

    return supp_transform, query_transform


def load(opt, logger, mode):
    split, shot, query = opt.split, opt.shot, 1
    height, width = opt.height, opt.width

    if mode == "train":
        data_transform = get_train_transforms(opt, height, width)
    elif mode in ["test", "eval_online", "predict"]:
        data_transform = get_val_transforms(opt, height, width)
    else:
        raise ValueError(f'Not supported mode: {mode}. [train|eval_online|test|predict]')

    if opt.dataset == "PASCAL":
        num_classes = 20
        cache = True
    elif opt.dataset == "COCO":
        num_classes = 80
        cache = False
    else:
        raise ValueError(f'Not supported dataset: {opt.dataset}. [PASCAL|COCO]')

    dataset = pfe.SemData(opt, split, shot, query,
                          data_root=DATA_DIR[opt.dataset],
                          data_list=DATA_LIST[opt.dataset][mode],
                          transform=data_transform,
                          mode=mode,
                          cache=cache)

    dataloader = DataLoader(dataset,
                            batch_size=opt.bs if mode == 'train' else opt.test_bs,
                            shuffle=True if mode == 'train' else False,
                            num_workers=opt.num_workers,
                            drop_last=True if mode == 'train' else False)

    logger.info(' ' * 5 + f"==> Data loader {opt.dataset} for {mode}")
    return dataset, dataloader, num_classes


def get_val_labels(opt, mode):
    if opt.dataset == "PASCAL":
        if opt.coco2pascal:
            if opt.split == 0:
                sub_val_list = [1, 4, 9, 11, 12, 15]
            elif opt.split == 1:
                sub_val_list = [2, 6, 13, 18]
            elif opt.split == 2:
                sub_val_list = [3, 7, 16, 17, 19, 20]
            elif opt.split == 3:
                sub_val_list = [5, 8, 10, 14]
            else:
                raise ValueError(f'PASCAL only have 4 splits [0|1|2|3], got {opt.split}')
        else:
            sub_val_list = list(range(opt.split * 5 + 1, opt.split * 5 + 6))
        return sub_val_list
    elif opt.dataset == "COCO":
        if opt.use_split_coco:
            return list(range(opt.split + 1, 81, 4))
        return list(range(opt.split * 20 + 1, opt.split * 20 + 21))
    else:
        raise ValueError(f'Only support datasets [PASCAL|COCO], got {opt.dataset}')
