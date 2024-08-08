import albumentations as A


distort = A.OneOf([
    A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    A.GridDistortion(p=1),
], p=0.15)


first_step = A.Compose([
    A.Rotate(p=0.5, limit=45),
    A.CoarseDropout(p=0.3, max_holes=30, max_height=32, max_width=32, min_holes=10, min_height=16, min_width=16),
    A.RandomBrightnessContrast(p=0.3, contrast_limit=0.4),
    distort,
])


hflip = A.Compose([
    A.HorizontalFlip(p=1),
    first_step
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))


vflip = A.Compose([
    A.VerticalFlip(p=1),
    first_step
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))


second_step = A.Compose([
    A.Rotate(p=0.8, limit=45),
    A.CoarseDropout(p=0.5, max_holes=30, max_height=32, max_width=32, min_holes=4, min_height=16, min_width=16),
    distort, 
])


darken = A.Compose([
    A.ToGray(p=1),
    second_step
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))


colorize = A.Compose([
    A.ColorJitter(p=1, brightness=0.2, contrast=0.3, saturation=0.3, hue=0.8),
    second_step
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))


crop = A.Compose([
    A.Resize(1024, 1024),
    A.OneOrOther(
        A.Compose([
            A.RandomSizedCrop(min_max_height=(256, 1024), height=512, width=512, p=1),
            A.OneOf([
                A.RandomSizedCrop(min_max_height=(256, 512), height=384, width=384, p=0.5),
                A.RandomSizedCrop(min_max_height=(256, 512), height=512, width=512, p=0.5),
            ]),
        ]),
        A.Compose([
            A.RandomSizedCrop(min_max_height=(256, 1024), height=384, width=384, p=1),
            A.OneOf([A.HueSaturationValue(p=0.5), A.RGBShift(p=0.7)], p=1),
        ]),
    ),
    A.HorizontalFlip(p=1),
    A.RandomBrightnessContrast(p=0.5),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
