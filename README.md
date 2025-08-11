# [CVPR'24/arXiv'25] Power Battery Detection

This repo is for the following two works:

**Power Battery Detection** [[Paper]](https://arxiv.org/pdf/2312.02528v2.pdf) 
<br>_Xiaoqi Zhao, Peiqian Cao, Lihe Zhang, Zonglei Feng, Hanqi Liu, Jiaming Zuo, Youwei Pang, Weisi Lin, Georges El Fakhri, Huchuan Lu, Xiaofeng Liu_<br>
In arXiv 2025 (T-PAMI under review)

**Towards Automatic Power Battery Detection:  New Challenge, Benchmark Dataset and Baseline** [[Paper]](https://arxiv.org/pdf/2312.02528v2.pdf) 
<br>_Xiaoqi Zhao, Youwei Pang, Zhenyu Chen, Qian Yu, Lihe Zhang, Hanqi Liu, Jiaming Zuo, Huchuan Lu_<br>
In CVPR 2024


### !! If you are interested in Ai4Industry, feel free to contact with us via Email (xiaoqi.zhao@yale.edu, lartpang@gmail.com)

## News
- Our extension work "**Power Battery Detection**" is available at [arXiv](https://arxiv.org/pdf/2312.02528v2.pdf).
  
## Motivation of Power Battery Detection
| PBD5K |
| --- |
| <img src="./image/motivation_PBD.png" width="800"/> |

 (a) EV sales are rapidly increasing, raising battery safety demands. (b) Battery packs account for ∼35%
 of EV cost, highlighting their critical role. (c) Assembly process of power batteries for EVs, where PBD is applied to the battery cells before assembly. (d)
 Illustration of the power battery detection task.
 
## Dataset
| PBD5K | Co-occurrence distribution of attributes |
| --- | --- |
| <img src="./image/PBD5K.png" width="400"/> | <img src="./image/PBD5K_1.png" width="400"/> |

|  Structural diversity | Attribute descriptions |
| --- | --- |
| <img src="./image/PBD5K_2.png" width="400"/> | <img src="./image/attribute_PBD5K.png" width="400"/> |

|  Visual examples from the PBD5K dataset |
| --- |
| <img src="./image/Visual_PBD5K.png" width="800"/> |


## Solution & Framework 
|   Multi-clue mining with point, line and number clues |
| --- |
| <img src="./image/solution.png" width="800"/> |

|   MDCNeXt |
| --- |
| <img src="./image/MDCNeXt.png" width="800"/> |

## Visual Comparison
|   Qualitative results on regular, difficult,and tough examples with varying attributes |
| --- |
| <img src="./image/Visual_result.png" width="800"/> |

|   Visual comparison with other  general/tinyobject detection-based, counting-based,  corner detection-based solutions|
| --- |
| <img src="./image/Visual_comparsion.png" width="800"/> |

|   Visual comparison with different image segmentation methods|
| --- |
| <img src="./image/Visual_comparsion_2.png" width="800"/> |

## Datasets Link
-  **PBD5K**: [GitHub Release](https://github.com/Xiaoqi-Zhao-DLUT/X-ray-PBD/releases/tag/Dataset)
-  **X-ray PBD (raw data)**: [Google Drive](https://drive.google.com/file/d/1d_b1V9XimIZSVVPxr3WlKaBtUsIGr9Mq/view?usp=sharing)
-  **X-ray PBD (training data processed in MDCNet)**: [Google Drive](https://drive.google.com/file/d/181Ct0wX05Wc5Ac_LCCgMO9q50vmdGky5/view?usp=sharing)

## Trained Model
-  You can download the trained MDCNeXt model at  [GitHub Release](https://github.com/Xiaoqi-Zhao-DLUT/X-ray-PBD/releases/tag/Model_pth)
-  You can download the trained PBD5K_Crop model at  [GitHub Release](https://github.com/Xiaoqi-Zhao-DLUT/X-ray-PBD/releases/tag/Model_pth)
-  You can download the trained MDCNet model at [Google Drive](https://drive.google.com/file/d/1NU0xWcRwipYkgj1YxMABoO-Kd3VRcdPU/view?usp=sharing)

Here’s your **infer** section rewritten in clear English, following a typical GitHub README style:

---

## **Inference**

1. **Configure the test dataset path**
   After downloading the dataset, edit `utils/config.py` to set the test dataset path:

   ```python
   # utils/config.py
   test_data_path = '/path/to/your/test_dataset'
   ```

2. **Configure the model checkpoint path**
   After downloading the pre-trained weights, edit `infer.py` to set the checkpoint path:

   ```python
   # infer.py
   ckpt_path = '/path/to/your/model_weights.pth'
   ```

3. **Install the `csrc` module**
   Download the `csrc` folder and place it in the project directory so that the structure looks like this:

   ```
   ├── model/
   │   └── MDCNeXt.py
   ├── csrc/
   │   ├── ...
   │   ├── ...
   │   └── setup.py
   ```

   Then run:

   ```bash
   cd csrc
   python setup.py install
   ```

4. **Run inference**
   Once the `csrc` module is installed, run:

   ```bash
   python infer.py
   ```


## Prediction Maps
-  You can download the prediction (crop_point_mask, location, original point mask) at  [GitHub Release](https://github.com/Xiaoqi-Zhao-DLUT/X-ray-PBD/releases/tag/Prediction_MDCNeXt)

Here’s the **Evaluation** section in clean README-style English:

---

## **Evaluation**

### **1. PBD Metrics**

1. Edit `config.py` to set the dataset root and model (prediction) path:

   ```python
   # config.py
   dataset_root_test = './'   # Path containing the test dataset
   Model = ''                 # prediction_path: directory containing both test dataset and predictions
   ```
2. Run:

   ```bash
   python test_score.py
   ```

---

### **2. Segmentation Metrics**

For segmentation evaluation, please use the official toolkit:
[**PySegMetric\_EvalToolkit**](https://github.com/Xiaoqi-Zhao-DLUT/PySegMetric_EvalToolkit)



## To Do List

- [x] Release data sets.
- [x] Release model code.
- [x] Release model weights.






## Citation

If you think X-ray-PBD codebase are useful for your research, please consider referring us:

```bibtex
@inproceedings{X-ray-PBD,
  title={Towards Automatic Power Battery Detection: New Challenge, Benchmark Dataset and Baseline},
  author={Zhao, Xiaoqi and Pang, Youwei and Chen, Zhenyu and Yu, Qian and Zhang, Lihe and Liu, Hanqi and Zuo, Jiaming and Lu, Huchua},
  booktitle={CVPR},
  year={2024}
```
