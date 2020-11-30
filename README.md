# 2020-SciChina-Eloss

The official pytorch implemention of the paper "Cognitive Vision Inspired Object Segmentation Metric and Loss Function"

## Introduction

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