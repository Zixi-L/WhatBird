import os
import gc
import time
import shutil
import random
import warnings
import typing as tp
from pathlib import Path
from contextlib import contextmanager

import yaml
from joblib import delayed, Parallel

import cv2
import librosa
import audioread
import soundfile as sf

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import resnest.torch as resnest_torch

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions as ppe_extensions
import pytorch_pfn_extras.training.extensions as extensions

import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display


pd.options.display.max_rows = 500
pd.options.display.max_columns = 500


# import five example files:
y_bulori, sr_bulori = librosa.load('data/bulori/XC128942.wav')
bulori, _ = librosa.effects.trim(y_bulori)

y_chispa, sr_chispa = librosa.load('data/chispa/XC16985.wav')
chispa, _ = librosa.effects.trim(y_chispa)

y_bkcchi, sr_bkcchi = librosa.load('data/bkcchi/XC16900.wav')
bkcchi, _ = librosa.effects.trim(y_bkcchi)

y_blujay, sr_blujay = librosa.load('data/blujay/XC16897.wav')
blujay, _ = librosa.effects.trim(y_blujay)

y_wilfly, sr_wilfly  = librosa.load('data/wilfly/XC31213.wav')
wilfly, _ = librosa.effects.trim(y_wilfly)


# play audio
def play_audio(file):
    x , sr = librosa.load(file)
    return ipd.Audio(file)


# Define utilities
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
#     torch.backends.cudnn.deterministic = True  # type: ignore
#     torch.backends.cudnn.benchmark = True  # type: ignore
    

@contextmanager
def timer(name: str) -> None:
    """Timer Util"""
    t0 = time.time()
    print("[{}] start".format(name))
    yield
    print("[{}] done in {:.0f} s".format(name, time.time() - t0))
    

# set strings
settings_str = 
"""
globals:
  seed: 1213
  device: cpu
  num_epochs: 20
  output_dir: training_output_v4/
  use_fold: 0
  target_sr: 32000

dataset:
  name: SpectrogramDataset
  params:
    img_size: 224
    melspectrogram_parameters:
      n_mels: 128
      fmin: 20
      fmax: 16000
    
split:
  name: StratifiedKFold
  params:
    n_splits: 5
    random_state: 42
    shuffle: True

loader:
  train:
    batch_size: 20
    shuffle: True
    num_workers: 0
    pin_memory: True
    drop_last: True
  val:
    batch_size: 20
    shuffle: False
    num_workers: 0
    pin_memory: True
    drop_last: False

model:
  name: resnest50_fast_1s1x64d
  params:
    pretrained: True
    n_classes: 5

loss:
  name: CrossEntropyLoss
  params: {}

optimizer:
  name: Adam
  params:
    lr: 0.001

scheduler:
  name: CosineAnnealingLR
  params:
    T_max: 10
"""

settings = yaml.safe_load(settings_str)



# read data

def train_audio_dir(file_path):
    return Path(file_path)



# preprocess audio data
def resample(ebird_code: str,filename: str, target_sr: int):    
    audio_dir = TRAIN_AUDIO_DIR
    resample_dir = TRAIN_RESAMPLED_DIR
    ebird_dir = resample_dir / ebird_code
    
    try:
        y, _ = librosa.load(
            audio_dir / ebird_code / filename,
            sr=target_sr, mono=True, res_type="kaiser_fast")

        filename = filename.replace(".mp3", ".wav")
        sf.write(ebird_dir / filename, y, samplerate=target_sr)
    except Exception as e:
        print(e)
        with open("skipped.txt", "a") as f:
            file_path = str(audio_dir / ebird_code / filename)
            f.write(file_path + "\n")

keys = set(train.ebird_code)
values = np.arange(0, len(keys))
BIRD_CODE  = dict(zip(sorted(keys), values))
INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}

PERIOD = 5

def mono_to_color(
    X: np.ndarray, mean=None, std=None,
    norm_max=None, norm_min=None, eps=1e-6
):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

class SpectrogramDataset(data.Dataset):
    def __init__(
        self,
        file_list: tp.List[tp.List[str]], img_size=224,
        waveform_transforms=None, spectrogram_transforms=None, melspectrogram_parameters={}
    ):
        self.file_list = file_list  # list of list: [file_path, ebird_code]
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        wav_path, ebird_code = self.file_list[idx]

        y, sr = sf.read(wav_path)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)
        else:
            len_y = len(y)
            effective_length = sr * PERIOD
            if len_y < effective_length:
                new_y = np.zeros(effective_length, dtype=y.dtype)
                start = np.random.randint(effective_length - len_y)
                new_y[start:start + len_y] = y
                y = new_y.astype(np.float32)
            elif len_y > effective_length:
                start = np.random.randint(len_y - effective_length)
                y = y[start:start + effective_length].astype(np.float32)
            else:
                y = y.astype(np.float32)

        melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)
        melspec = librosa.power_to_db(melspec).astype(np.float32)

        if self.spectrogram_transforms:
            melspec = self.spectrogram_transforms(melspec)
        else:
            pass

        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

#         labels = np.zeros(len(BIRD_CODE), dtype="i")
        labels = np.zeros(len(BIRD_CODE), dtype="f")
        labels[BIRD_CODE[ebird_code]] = 1

        return image, labels
    
# Train Utility

def get_loaders_for_training(
    args_dataset: tp.Dict, args_loader: tp.Dict,
    train_file_list: tp.List[str], val_file_list: tp.List[str]
):
    # # make dataset
    train_dataset = SpectrogramDataset(train_file_list, **args_dataset)
    val_dataset = SpectrogramDataset(val_file_list, **args_dataset)
    # # make dataloader
    train_loader = data.DataLoader(train_dataset, **args_loader["train"])
    val_loader = data.DataLoader(val_dataset, **args_loader["val"])
    return train_loader, val_loader

def get_model(args: tp.Dict):
    model =getattr(resnest_torch, args["name"])(pretrained=args["params"]["pretrained"])
    del model.fc
    # # use the same head as the baseline notebook.
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, args["params"]["n_classes"]))
    
    return model

def train_loop(
    manager, args, model, device,
    train_loader, optimizer, scheduler, loss_func
):
    """Run minibatch training loop"""
    while not manager.stop_trigger:
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            with manager.run_iteration():
                data, target = data.to(device), target.to(device)
                target = torch.argmax(target, axis=-1)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_func(output, target) # Loss: CrossEntropy
                ppe.reporting.report({'train/loss': loss.item()})
                loss.backward()
                optimizer.step()
        scheduler.step()  # <- call at the end of each epoch
        

def eval_for_batch(
    args, model, device,
    data, target, loss_func, eval_func_dict={}
):
    """
    Run evaliation for valid
    
    This function is applied to each batch of val loader.
    """
    model.eval()
    data, target = data.to(device), target.to(device)
    output = model(data)
    # Final result will be average of averages of the same size
    target = torch.argmax(target, axis=-1) # From 2D to 1D
    val_loss = loss_func(output, target).item()
    ppe.reporting.report({'val/loss': val_loss})
    
    for eval_name, eval_func in eval_func_dict.items():
        eval_value = eval_func(output, target).item()
        ppe.reporting.report({"val/{}".format(eval_aame): eval_value})
        
    print(output.shape, target.shape)


def set_extensions(
    manager, args, model, device, test_loader, optimizer, evaluator,
    loss_func, eval_func_dict={}
):
    """set extensions for PPE"""
        
    my_extensions = [
        # # observe, report
        ppe_extensions.observe_lr(optimizer=optimizer),
        # ppe_extensions.ParameterStatistics(model, prefix='model'),
        # ppe_extensions.VariableStatisticsPlot(model),
        ppe_extensions.LogReport(),
        ppe_extensions.IgniteEvaluator(
            evaluator, test_loader, model, progress_bar=True),
        ppe_extensions.PlotReport(['train/loss', 'val/loss'], 'epoch', filename='loss.png'),
        ppe_extensions.PlotReport(['lr',], 'epoch', filename='lr.png'),
        ppe_extensions.PrintReport([
            'epoch', 'iteration', 'lr', 'train/loss', 'val/loss','val/acc', "elapsed_time"]),
#         ppe_extensions.ProgressBar(update_interval=100),

        # # evaluation
        (
            ppe_extensions.Evaluator(
                test_loader, model,
                eval_func=lambda data, target:
                    eval_for_batch(args, model, device, data, target, loss_func, eval_func_dict),
                progress_bar=True),
            (1, "epoch"),
        ),
        # # save model snapshot.
        (
            ppe_extensions.snapshot(
                target=model, filename="snapshot_epoch_{.updater.epoch}.pth"),
            ppe.training.triggers.MinValueTrigger(key="val/loss", trigger=(1, 'epoch'))
        ),
    ]
           
    # # set extensions to manager
    for ext in my_extensions:
        if isinstance(ext, tuple):
            manager.extend(ext[0], trigger=ext[1])
        else:
            manager.extend(ext)
        
    return manager
    

# Train
# get wav file path

train=train.rename(columns={'filename':'resampled_filename'})
train.resampled_filename=[filename.replace(".mp3", ".wav") for filename in train.resampled_filename]

tmp_list = []
for ebird_d  in TRAIN_RESAMPLED_AUDIO_DIRS.iterdir():
    if os.path.split(ebird_d)[1]=='.DS_Store':
        continue

    for wav_f in ebird_d.iterdir():
        tmp_list.append([ebird_d.name, wav_f.name, wav_f.as_posix()])

train_wav_path_exist = pd.DataFrame(
    tmp_list, columns=["ebird_code", "resampled_filename", "file_path"])

del tmp_list

train_all = pd.merge(train, train_wav_path_exist, on=["ebird_code", "resampled_filename"], how="inner")

# split data
skf = StratifiedKFold(**settings["split"]["params"])

train_all["fold"] = -1
for fold_id, (train_index, val_index) in enumerate(skf.split(train_all, train_all["ebird_code"])):
    train_all.iloc[val_index, -1] = fold_id
    
# # check the propotion
fold_proportion = pd.pivot_table(train_all, index="ebird_code", columns="fold", values="xc_id", aggfunc=len)

use_fold = settings["globals"]["use_fold"]
train_file_list = train_all.query("fold != @use_fold")[["file_path", "ebird_code"]].values.tolist()
val_file_list = train_all.query("fold == @use_fold")[["file_path", "ebird_code"]].values.tolist()


# running training 

set_seed(settings["globals"]["seed"])
device = torch.device(settings["globals"]["device"])
output_dir = Path(settings["globals"]["output_dir"])

# # # get loader
train_loader, val_loader = get_loaders_for_training(
    settings["dataset"]["params"], settings["loader"], train_file_list, val_file_list)

# # # get model
model = get_model(settings["model"])
model = model.to(device)

# # # get optimizer
optimizer = getattr(
    torch.optim, settings["optimizer"]["name"]
)(model.parameters(), **settings["optimizer"]["params"])

# # # get scheduler
scheduler = getattr(
    torch.optim.lr_scheduler, settings["scheduler"]["name"]
)(optimizer, **settings["scheduler"]["params"])

# # # get loss
loss_func = getattr(nn, settings["loss"]["name"])(**settings["loss"]["params"])

# get evaluator
evaluator = create_supervised_evaluator(
    model,
    metrics={'acc': Accuracy(is_multilabel=True)},
    device=device,
    output_transform=lambda x,y,y_pred: (torch.FloatTensor(y_pred.shape).zero_().scatter_(0,  torch.argmax(y_pred, 0,keepdim=True), 1),
                                         y))

      
# # # create training manager
trigger = None


manager = ppe.training.ExtensionsManager(
    model, optimizer, settings["globals"]["num_epochs"],
    iters_per_epoch=len(train_loader),
    stop_trigger=trigger,
    out_dir=output_dir
)

# # # set manager extensions
manager = set_extensions(
    manager, settings, model, device,
    val_loader, optimizer,evaluator ,loss_func,
)

# Train loop

# # runtraining with lr = 0.001

train_loop(
    manager, settings, model, device,
    train_loader, optimizer, scheduler, loss_func)