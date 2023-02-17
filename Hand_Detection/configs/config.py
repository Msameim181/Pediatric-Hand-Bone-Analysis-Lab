from datetime import datetime

from mmcv import Config
from mmdet.apis import set_random_seed
from model_downloader import download_model

start_time = datetime.now()
run_name = start_time.strftime("%Y%m%d_%H%M%S")

cfg = Config.fromfile('./YOLOX/yoloxs.py')

cfg.dataset_type = 'CocoDataset'
cfg.data_root = 'data/hands/'
classes = ('hand',)

cfg.data.train.dataset.img_prefix=f'{cfg.data_root}/images/'
cfg.data.train.dataset.classes=classes
cfg.data.train.dataset.ann_file=f'{cfg.data_root}/train.json'

cfg.data.val.img_prefix=f'{cfg.data_root}/images/'
cfg.data.val.classes=classes
cfg.data.val.ann_file=f'{cfg.data_root}/val.json'

cfg.data.test.img_prefix=f'{cfg.data_root}/images/'
cfg.data.test.classes=classes
cfg.data.test.ann_file=f'{cfg.data_root}/val.json'

model_url = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
cfg.load_from = 'yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
download_model(cfg.load_from, model_url)

# Set up working dir to save files and logs.
cfg.max_epochs = 300
cfg.device = "cuda"
cfg.work_dir = f'./{run_name}'
# cfg.resume_from = 'hand_detect_368/best_bbox_mAP_epoch_100.pth'

# set log interval
cfg.log_config.interval = 1
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='MMDetWandbHook',
         init_kwargs={'project': 'Pediatric_Hand_Detection',
                      'entity': 'msameim181',
                      'name': run_name},
         interval=10,
         log_checkpoint=True,
         log_checkpoint_metadata=True,
         num_eval_images=100)
]
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
print(f'Config:\n{cfg.pretty_text}')