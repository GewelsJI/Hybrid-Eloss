# Hybrid Eloss for object segmentation in PyTorch

The official pytorch implemention of the paper "Cognitive Vision Inspired Object Segmentation Metric and Loss Function"

## Introduction

<p align="center">
    <img src="imgs/D-measureFramewrok.png"/> <br />
    <em> 
    Figure 1: The pipeline of our $E_\xi$.
  (a) ground-truth （GT）map. (b) the binary foreground map.
  (c) \& (d) are the mean values map of GT \& FM, respectively.
  (e) and (f) are the bias matrices calculated by Eqn.~\ref{equ:bias_matrix}.
  (g) is the mapping function.
  (h) is the enhanced alignment matrix computed by Eqn.~\ref{equ:enhance_alignment_matrix}.
  \emph{``aligned''} \& \emph{``unaligned''} donate those points which
  $GT(x,y) = FM(x,y)$ \& $GT(x,y) \neq FM(x,y)$, respectively.
    </em>
</p>

Object segmentation (OS) technology is a research hotspot in computer vision, and it has a wide range of applications 
in many fields. Cognitive vision studies have shown that human vision is highly sensitive to both global information 
and local details in scenes.
To this end, we design a novel, efficient, and easy-to-use Enhanced-alignment measure ($E_\xi$) for evaluating 
the performance of the OS model.
$E_\xi$ combines local pixel values with the image-level mean value, jointly evaluate the image-/pixel-level similarity 
between a segmentation result and a ground-truth (GT) results.
Extensive experiments on the four popular benchmarks via five meta-measures, i.e., application ranking, demoting generic, 
denying noise, human ranking, and recognizing GT, we observe significant relative improvement compared with existing 
widely-adopted evaluation metrics such as IoU and $F_\beta$.
By using the weighted binary cross-entropy loss, the Enhanced-alignment loss, and the weighted IoU loss, we further 
design a hybrid loss function (Hybrid-$E_{loss}$) to guide the network to learn pixel-, object- and image-level features.
Qualitative and quantitative results show further improvement in terms of accuracy when using our hybrid loss function 
in three different OS tasks.

## Usage

Here, we provide a toy demo of our Hybrid Eloss.

```python
# -*- coding: utf-8 -*-
import torch
import
from Hybrid_Eloss import hybrid_e_loss

# set the hyper-parameters
bs, c, w, h = 2, 1, 352, 352
learning_rate = 1e-6
epoch = 100

# get your prediction map with CUDA mode
pred = torch.randn(bs, c, w, h).cuda() # bs, c, w, h
# get your ground-truth mask with CUDA mode
gt = torch.randn(bs, c, w, h).cuda()


for i in range(epoch):
    # 1. define forward pass
    pred = model(x)
    # 2. compute loss
    loss = hybrid_e_loss(pred, gt)
    # 3. backprop and update weights
    loss.backward()
```
## Applications

To verify the effectiveness of our Hybrid Eloss, we employ it to existing task in the binary segmentation tasks, including Salient Object detection (SOD), Camouflaged Object Segmentation (COD), and Polyp Segmentation (PSeg).

<p align="center">
    <img src="imgs/visual_results-min.png"/> <br />
    <em> 
    Figure 2: Visual comparison of the proposed Hybrid-$E_{loss}$ function via decoupling it, \textit{i.e.}, Hybrid-$E_{loss} = \mathcal{L}^{w}_{ce} + \mathcal{L}^{w}_{iou} + \mathcal{L}_{e}$, we demonstrate the effectiveness of three sub-variants and our $E_{loss}$ on three typical object segmentation tasks, including salient object detection (SOD, \ie, SCRN), camouflaged object detection (COD, \ie, SINet) and polyp segmentation (Polyp Seg., \ie, PraNet).
    </em>
</p>

### Task-1: Salient Object detection

SCRN

- Data Preparation
  
  Download the training ([Cowtransfer Drive]()) and tetsting ([Cowtransfer Drive]()) set and put it into `./data/SOD/`.

- Training
  
  Choose the differnt type of loss function in  and just run `python ./scripts/SCRN/training.py`

- Testing
  
  Download the pretrained weights ([Cowtransfer Drive]()) and put them into `./snapshots/SCRN/`.

- Results

### Task-2: Camouflaged Object Segmentation

SINet

- Data Preparation
  
  Download the training ([Cowtransfer Drive]()) and tetsting ([Cowtransfer Drive]()) set and put it into `./data/COS/`.

- Training
- Testing
  
  Download the pretrained weights ([Cowtransfer Drive]()) and put them into `./snapshots/SINet/`.

- Results

### Task-3: Polyp Segmentation

- Data Preparation
  
  Download the training ([Cowtransfer Drive]()) and tetsting ([Cowtransfer Drive]()) set and put it into `./data/PSeg/`.

- Training
- Testing
  
  Download the pretrained weights ([Cowtransfer Drive]()) and put them into `./snapshots/PraNet/`.

- Results