#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# inference of model 45 (best model so far), similar to 53a but with on-the-fly dict processing


# In[ ]:


import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""


# In[ ]:


from fastai.basics           import *
from fastai.callback.all     import *
from fastai.vision.all       import *
from fastai.medical.imaging  import *
from fastai.distributed      import *

import torchvision.transforms as transforms

import multiprocessing
import pydicom
import pandas as pd
import shutil

from collections import defaultdict
from pydicom.pixel_data_handlers.util import apply_modality_lut
from tqdm.notebook import tqdm
from pathlib import Path


# # Hyperparams

# In[ ]:


cpus = multiprocessing.cpu_count()
cpus


# In[ ]:


class H:
    '''Hyperparams'''
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def __repr__(self):
        return str(self.__dict__)


# In[ ]:


h = H(
    model_name = '45',
    size       = 256,
    bs         = 16,
    window     = (700, 100),  # ww, wc (1600, 550)
    study_len  = 63,
    slices     = 5,
    pool       = (1, 3),
    tfm_ftrs   = 128, # transformer input feature len
    n_tta      = 5,
)


# # Env prep

# In[ ]:


KAGGLE = Path('/kaggle').exists()


# ## Env dependent paths

# In[ ]:


if KAGGLE:
    # Use kaggle test sets
    ds_dir     = Path('/kaggle/input/rsna-str-ped')
    test_dir   = Path('/kaggle/input/rsna-str-pulmonary-embolism-detection')
else:
    ds_dir     = Path('kaggle_dataset/to_upload')
    test_dir   = Path('/space/sped')


# ## Unpack dataset

# In[ ]:


if Path('resources').exists():
    shutil.rmtree('resources')
    
for tgz in ds_dir.glob('*.tgz'):
    tgz = ds_dir / 'resources.tgz'
    assert os.system(f'tar xvf {str(tgz)}') == 0


# ## Install gdcm deps
# These are required by pydicom

# In[ ]:


if KAGGLE:
    assert os.system('conda install --offline resources/gdcm/gdcm-2.8.9-py37h71b2a6d_0.tar.bz2') == 0


# ## Install wheels

# In[ ]:


if KAGGLE:
    assert os.system('pip install resources/wheels/*') == 0


# In[ ]:


from positional_encodings import PositionalEncoding3D
from reformer_pytorch import Reformer, Autopadder


# # Data

# In[ ]:


study_labels = [
    # useful [:9]
    'negative_exam_for_pe', 
    'indeterminate',
    'chronic_pe', 
    'acute_and_chronic_pe', 
    'central_pe', 
    'leftsided_pe', 
    'rightsided_pe', 
    'rv_lv_ratio_gte_1', 
    'rv_lv_ratio_lt_1',
    # ancillary [9:]
    'qa_motion', 
    'qa_contrast', 
    'flow_artifact', 
    'true_filling_defect_not_pe',
]


# In[ ]:


min_cols_i_want = [
    'ImagePositionPatient',
    'StudyInstanceUID',
    'SeriesInstanceUID',
    'SOPInstanceUID',
]


# In[ ]:


def tabulate_dicom_data(file_list):
    l = []
    for f in file_list:
        try:
            dicom = pydicom.dcmread(str(f))
            dicom.pixel_array # try to get this
            d = {}
            for s in min_cols_i_want:
                if dicom.data_element(s) is not None:
                    v = dicom.data_element(s).value
                    if isinstance(v, pydicom.multival.MultiValue):
                        for i in range(len(v)):
                            d[s + '_' + str(i)] = v[i]
                    else:
                        d[s] = v
            l.append(d)
        except:
            pass
        
    return l


# In[ ]:


files = get_dicom_files(test_dir / 'test/')


# In[ ]:


if len(files) == 146853:
    files = files[:1024]


# In[ ]:


if not Path('inference_df.fth').exists():
    with ProcessPoolExecutor(max_workers=cpus) as e:
         l = list(tqdm(itertools.chain.from_iterable(e.map(tabulate_dicom_data, np.array_split(files, 32))), total=len(files)))
    df = pd.DataFrame(l).sort_values(['StudyInstanceUID', 'ImagePositionPatient_2']).reset_index(drop=True)
    df.to_feather('inference_df.fth')
else:
    df = pd.read_feather('inference_df.fth')


# In[ ]:


dfg = df.groupby('StudyInstanceUID')


# In[ ]:


l = []
for g in dfg.groups:
    group = dfg.get_group(g)
    new_group = group.copy()
    n_subgroups = (len(group) + h.slices * h.study_len - 1) // (h.slices * h.study_len)
    n_repeats = (len(group) + n_subgroups - 1) // n_subgroups
    new_group['subgroup'] = np.tile(range(n_subgroups), n_repeats)[:len(group)]
    l.append(new_group)
df = pd.concat(l)


# In[ ]:


dfg = df.groupby(['StudyInstanceUID', 'subgroup'])


# In[ ]:


# to show batch way 2: make DicomTransform derive from Transform, so that its 'decode' gets called twice:
# one for x which returns a TensorImage, which is showable, and another for y which returns this:
class TitledTensor(TensorBase):
    def show(self, **kwargs): show_title(self, **kwargs)


# In[ ]:


class DicomReader:
    def __init__(self, dfg):
        self.dfg = dfg
        # keep in a dict the first and last indices of each study
        self.studies = list(dfg.indices.keys())
        self.cache = {}

    def get_slices(self, group, i, slices):
        ixs = np.arange(i - slices // 2, i - slices // 2 + slices).clip(0, len(group) - 1)
        rows = group.iloc[ixs]
        return rows
        
    def get_image(self, group, i, slices, clear_cache=True):
        x = torch.zeros(slices, 512, 512, dtype=torch.uint8) # (C, H, W)
        rows = self.get_slices(group, i, slices)
        #sop_uids = np.empty([slices], dtype=np.dtype('U12'))
        sop_uids = []
        for j, (idx, row) in enumerate(rows.iterrows()):
            if row.SOPInstanceUID not in self.cache:
                p = test_dir / 'test' / row.StudyInstanceUID / row.SeriesInstanceUID / f'{row.SOPInstanceUID}.dcm'
                dcm = pydicom.dcmread(str(p))
                a = apply_modality_lut(dcm.pixel_array, dcm).astype(np.float32) # TODO: to float32 if memory runs out?
                t = torch.from_numpy(a)
                if t.shape != (384, 384): 
                    t = t.view(1, 1, *t.shape) # HW -> BCHW
                    t = torch.nn.functional.interpolate(t, size=512, mode='nearest') # resize (FloatTensors only!)
                    t = t.view(t.shape[-2:]) # BCHW -> HW
                ww, wc = h.window
                w0, w1 = wc-ww/2, wc+ww/2
                t = torch.clamp(t, w0, w1) # window
                t = (t - w0) * 255. / ww # normalize
                self.cache[row.SOPInstanceUID] = t.to(torch.uint8)
            x[j] = self.cache[row.SOPInstanceUID]
            #sop_uids[j] = row.SOPInstanceUID
            sop_uids.append(row.SOPInstanceUID)
            
        if clear_cache:
            self.cache = {}

        return x, sop_uids

    def get_center_img_indices(self, group, n_imgs, slices):
        idxs = np.linspace(slices // 2, len(group) - slices // 2 - 1, n_imgs).round().astype(np.int)
        return idxs
    
    def get_study(self, i, n_imgs, slices):
        self.cache = {}
        
        group = self.dfg.get_group(self.studies[i])
        study_uid = self.studies[i][0]
        
        x = torch.zeros(n_imgs, slices, 512, 512, dtype=torch.uint8) # (SL, C, H, W) SL=study len
        sop_uids_list = []

        idxs = self.get_center_img_indices(group, n_imgs, slices)
        for i, idx in enumerate(idxs):
            x[i], sop_uids = self.get_image(group, idx, slices, clear_cache=False)
            sop_uids_list.append(sop_uids)
            
        self.cache = {}

        return [TensorImage(x), FloatTensor([0.]), study_uid, sop_uids_list]


# In[ ]:


class StudyTransform(Transform):
    def __init__(self, dfg, n_imgs, slices, is_valid=False):
        self.dicom_reader = DicomReader(dfg)
        self.n_imgs = n_imgs
        self.slices = slices
        
    def encodes(self, i):
        return self.dicom_reader.get_study(i, self.n_imgs, self.slices)


# In[ ]:


class CenterCrop(ItemTransform):
    order=44

    def encodes(self, xy):
        x, y = xy
        x = x[:,:,64:-64,64:-64]
        return x, y        


# # Learner

# In[ ]:


#pos_weight = FloatTensor([5.]).cuda()

def loss_func(pred, targ):
    #print('pred:', pred.shape) # pred: torch.Size([128, 1])
    #print('targ[0]:', targ[0].shape) # targ: torch.Size([1, 128])
    #print('targ[1]:', targ[1].shape) # targ: torch.Size([1, 128])
    #return BCEWithLogitsLossFlat(pos_weight=pos_weight)(pred, targ)
    return torch.zeros([]).cuda()


# In[ ]:


def petaformer_splitter(model):
    return [backbone_params, head_params]


# In[ ]:


class FiveD2FourD(ItemTransform):
    order=5
    def encodes(self, xy):
        x, y = xy
        b, sl, c, h, w = x.shape
        return x.view(b * sl, c, h, w), y # B*SL, C, H, W


# In[ ]:


class FourD2FiveD(ItemTransform):
    order=45
    def encodes(self, xy):
        x, y = xy
        bsl, c, he, wi = x.shape
        return x.view(1, bsl, c, he, wi), y # B, SL, C, H, W


# In[ ]:


class ApplyTransformsToXYAndBypassEverythingElse(ItemTransform):
    def __init__(self, tfms):
        self.tfms = tfms
        
    def encodes(self, xyplus):
        x, y, *everything_else = xyplus
        x, y = self.tfms((x, y))
        return (x, y, *everything_else)


# In[ ]:


class MLPModel(nn.Module):
    def __init__(self, f1, f2, f3, f4, dropout=0.25):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(f1, f2),
            nn.ReLU(),
#            nn.BatchNorm1d(f2),
            nn.Dropout(dropout),            
            nn.Linear(f2, f3),
            nn.ReLU(),
#            nn.BatchNorm1d(f3),
            nn.Dropout(dropout),
            nn.Linear(f3, f4),
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


# In[ ]:


class PetaFormer(Module):
    def __init__(self, backbone, pool):
        self.backbone = backbone
        self.fc_pos_enc = nn.Linear(512, h.tfm_ftrs)
        self.pos3d = PositionalEncoding3D(h.tfm_ftrs)
        
        bfs = pool[0] * pool[1] # backbone featuremap size
        
        self.transformer = Autopadder(Reformer(
            dim = h.tfm_ftrs,
            depth = 6,
            max_seq_len = h.study_len * bfs + 1,
            heads = 8,
            lsh_dropout = 0.1,
            causal = False,
        ))
        #self.transformer = LinearAttentionTransformer(
        #    dim = 512,
        #    heads = 8,
        #    depth = 4,
        #    max_seq_len = h.study_len * 16 + 1, # max slices * backbone activation map dims
        #    n_local_attn_heads = 4
        #)
        
        self.avgpool2d = nn.AdaptiveAvgPool2d(output_size=pool)
        #self.avgpool1d = nn.AdaptiveAvgPool1d(output_size=1)
        self.flatten = Flatten()
        #self.fc = nn.Linear(512, 1)
        self.mlp_image = MLPModel(h.tfm_ftrs * bfs, h.tfm_ftrs * bfs // 2, h.tfm_ftrs * bfs // 4, h.slices)
        self.mlp_study = MLPModel(h.tfm_ftrs, h.tfm_ftrs // 2, h.tfm_ftrs // 4, 13)

        
    
    def forward(self, x):
        bs, sl, c, h, w = x.shape
        x = x.view(bs * sl, c, h, w)
        
        # Backbone
        y = self.backbone(x) # B * SL, 512, FH, FW (FH=FW=4 for h.size up to 128)
        y = self.avgpool2d(y) # B * SL, 512, 1, 7
        fh, fw = y.shape[2:]
        
        # 3D positional encoding
        y = y.permute(0, 2, 3, 1) # B * SL, FH, FW, 512
        y = y.view(bs, sl, *y.shape[-3:]) # B, SL, FH, FW, 512
        y = self.fc_pos_enc(y) # B, SL, FH, FW, TF (TF=h.tfm_ftrs)
        tf = y.shape[-1]
        y = y + self.pos3d(y) # B, SL, FH, FW, TF

        # Transformer
        y = y.reshape(bs, sl * fh * fw, tf) # B, SL * FH * FW, TF
        #print('into transformer:', y.shape)
        #pdb.set_trace()
        z = torch.zeros([bs, 1, tf], device=x.device) # B, 1, TF
        y = torch.cat([z, y], dim=1) # B, SL * FH * FW + 1, TF
        y = self.transformer(y) # B, SL * FH * FW + 1, TF
        #print('out of transformer:', y.shape)
        
        # Study pred
        yst = y[:,0] # B, 512
        #print(f'yst={yst.shape}')
        yst = self.mlp_study(yst) # B, 13
        
        # Image preds
        yim = y[:,1:] # B, SL * FH * FW, TF
        yim = yim.view(bs * sl, fh * fw * tf) # B * SL, FH * FW * TF
        #print(f'yim={yim.shape}')
        yim = self.mlp_image(yim) # B * SL, 3
        yim = yim.view(bs, sl, c) # B, SL, 3

        
        #print('preds:', y.shape)
        
        return (yim, yst)

    


# ## Load model

# In[ ]:


learn = load_learner(f'resources/{h.model_name}.pkl')


# In[ ]:


# WTF: aug_transforms' rotations were still applied on the test set
#augs = [Flip(p=0., size=h.size, batch=True)] # dummy p=0. flip works as a resizer
augs = aug_transforms(mode='nearest', size=h.size, do_flip=False, batch=True)


# In[ ]:


after_batch = Pipeline([
    ApplyTransformsToXYAndBypassEverythingElse(
        Pipeline([
            IntToFloatTensor, 
            FiveD2FourD,
            *augs,
            FourD2FiveD,
        ]),
    )
])


# In[ ]:





# In[ ]:


test_ds = StudyTransform(dfg, h.study_len, h.slices)


# In[ ]:


test_tl = TfmdLists(range(len(test_ds.dicom_reader.studies)), test_ds)


# In[ ]:


test_dl = TfmdDL(test_tl, bs=1, shuffle=False)


# In[ ]:


#tdl = learn.dls.test_dl(test_tl, bs=h.bs, num_workers=cpus)


# In[ ]:


learn.model.to(torch.device('cuda'))
test_dl.to(torch.device('cuda'))


# In[ ]:


study_dict = defaultdict(list) 
image_dict = defaultdict(list) 

learn.model.eval()
with torch.no_grad():
    for raw_batch in tqdm(test_dl):
        for i in range(h.n_tta):
            batch = after_batch(raw_batch)
            
            x = batch[0] # B, SL, C, W, H
            x.cuda()
            study_uids = batch[2] # B
            sop_uids = batch[3] # SL, C, B (nested lists)
        
            
            y = learn.model(x)

            #yim = y[:,:h.study_len*h.slices]
            #yim = yim.view(yim.shape[0], h.study_len, h.slices) # B, SL, C
            #yst = y[:,h.study_len*h.slices:]

            yim = y[0] # B, SL, C
            yst = y[1] # B, 13
            yst = yst[:,:9] # B, 9

            yim = torch.sigmoid(yim) # B, SL * C
            yst = torch.sigmoid(yst) # B, ST_LABELS


            for b in range(yst.shape[0]):
                study_uid = study_uids[b]

                # study preds
                for j in range(9):
                    study_dict[f'{study_uid}_{study_labels[j]}'].append(yst[b,j].item())

                # image preds
                for sl in range(yim.shape[1]):
                    for c in range(yim.shape[2]):
                        image_dict[sop_uids[sl][c][b]].append(yim[b,sl,c].item())


# In[ ]:


sub_list = []

for study_uid_and_label, preds_list in study_dict.items():
    study_tta_pred = sum(preds_list) / len(preds_list)
    sub_list.append([study_uid_and_label, study_tta_pred])

for sop_uid, preds_list in image_dict.items():
    sop_tta_pred = sum(preds_list) / len(preds_list)
    sub_list.append([sop_uid, sop_tta_pred])


# In[ ]:


sub_df = pd.DataFrame(sub_list, columns=['id', 'label']).set_index('id', drop=True)


# In[ ]:


sub_df['label'] = sub_df['label'].clip(0.001, 0.999)


# In[ ]:


sample_sub_df = pd.read_csv(test_dir / 'sample_submission.csv', index_col='id')


# In[ ]:


sample_sub_df.update(sub_df)


# In[ ]:


if KAGGLE:
    sample_sub_df.to_csv('submission.csv')
else:
    sample_sub_df.to_csv(f'submission_{h.model_name}.csv')


# In[ ]:




