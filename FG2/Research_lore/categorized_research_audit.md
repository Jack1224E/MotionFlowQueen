# Research Audit: High-Fidelity Link Extraction

This document contains a categorized map of every research paper and repository cited in the project's LLM reports, including items from the "Theoretical" report and user-verified additions.

## 1. Video Frame Interpolation Architectures

### Fused / Single-Pass Models
*(Focus: Latency minimization, unified flow+synthesis)*

- **IFRNet (Intermediate Feature Refine Network)**
  - **Repo:** `https://github.com/ltkong218/IFRNet`
  - **Paper:** [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Kong_IFRNet_Intermediate_Feature_Refine_Network_for_Efficient_Frame_Interpolation_CVPR_2022_paper.html)
  - *Context:* Jointly refines intermediate optical flow and context features. Key candidate for real-time deployment.

- **AMT (All-Pairs Multi-Field Transforms)**
  - **Repo:** `https://github.com/MCG-NKU/AMT`
  - **Paper:** [Arxiv 2023](https://arxiv.org/pdf/2304.09790.pdf)
  - *Context:* Uses bidirectional correlation volumes and multi-field candidates to handle occlusions in a single pass.

- **UPR-Net (Unified Pyramid Recurrent Network)**
  - **Repo:** `https://github.com/srcn-ivl/UPR-Net`
  - **Paper:** [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Jin_A_Unified_Pyramid_Recurrent_Network_for_Video_Frame_Interpolation_CVPR_2023_paper.pdf)
  - *Context:* Lightweight pyramid architecture suitable for iterative refinement.

- **LADDER (Efficient Framework)**
  - **Paper:** [Arxiv 2024](https://arxiv.org/pdf/2404.11108.pdf)
  - *Status:* `[MISSING REPO URL]`

- **BiM-VFI (Bidirectional Motion Field-Guided)**
  - **Paper:** [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Seo_BiM-VFI_Bidirectional_Motion_Field-Guided_Frame_Interpolation_for_Video_with_Non-uniform_CVPR_2025_paper.pdf)
  - *Status:* `[MISSING REPO URL]`

- **NAS-VFI (Neural Architecture Search)**
  - **Paper:** [Arxiv](https://arxiv.org/abs/2506.01061)
  - *Status:* `[MISSING REPO URL]`

- **VFI_Adapter (Boost Video Frame Interpolation via Motion Adaptation)**
  - **Repo:** `https://github.com/MCG-NKU/VFI_Adapter`
  - *Context:* Adaptive plug-in wrapper for pre-trained models.

- **GFFE (G-buffer Free Frame Extrapolation)**
  - *Status:* `[MISSING REPO URL]`
  - *Context:* Low-latency real-time rendering focus.

### Cascaded / Refinement Models
*(Focus: Quality benchmarks, Teacher models)*

- **RIFE (Real-Time Intermediate Flow Estimation)**
  - **Repo:** `https://github.com/hzwer/ECCV2022-RIFE`
  - **Paper:** [ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740608.pdf)
  - *Context:* The standard baseline for real-time VFI.

- **CAIN (Channel Attention Is All You Need)**
  - **Repo:** `https://github.com/myungsub/CAIN`
  - **Paper:** [Arxiv 2020](https://arxiv.org/abs/2004.11364)

- **EMA-VFI (Extracting Motion and Appearance)**
  - **Repo:** `https://github.com/MCG-NKU/EMA-VFI`
  - **Paper:** [Arxiv 2023](https://arxiv.org/abs/2304.02818)

- **VFIFormer**
  - **Repo:** `https://github.com/dvlab-research/vfiformer`
  - *Context:* Transformer-based, likely too heavy for real-time but good for distillation.

- **M2M-VFI (Many-to-Many Splatting)**
  - **Repo:** `https://github.com/feinanshan/M2M_VFI`

- **PerVFI (Perception-Oriented)**
  - **Repo:** `https://github.com/mulns/PerVFI`
  - **Paper:** [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Perception-Oriented_Video_Frame_Interpolation_via_Asymmetric_Blending_CVPR_2024_paper.pdf)

- **VFIMamba**
  - **Repo:** `https://github.com/MCG-NJU/VFIMamba`

## 2. Optical Flow & Motion Heuristics

### Hardware / Accelerated Flow
- **NVIDIA Optical Flow SDK (NVOFA)**
  - **Landing:** `https://developer.nvidia.com/optical-flow-sdk`
  - **Repository:** `https://github.com/NVIDIA/NVIDIAOpticalFlowSDK`
  - **Documentation:** `https://docs.nvidia.com/video-technologies/optical-flow-sdk/index.html`

- **OpenCV CUDA Optical Flow**
  - **Docs:** `https://docs.opencv.org/4.x/d7/d3f/group__cudaoptflow.html`

- **HighFPSViewer-NvOFFRUC (An Efficient Frame Interpolation Approach Based on NVIDIA Optical Flow SDK)**
  - **Repo:** `https://github.com/NVIDIA/HighFPSViewer-NvOFFRUC` (Inferred)
  - *Context:* Reference implementation utilizing DXGI desktop duplication and NvOFFRUC.

### Efficient / Learned Flow (Teachers)
- **LiteFlowNet**
  - **Repo:** `https://github.com/twhui/LiteFlowNet`
  - **Paper:** [CVPR 2018](https://openaccess.thecvf.com/content_CVPR_2018/papers/Hui_LiteFlowNet_A_Lightweight_CVPR_2018_paper.pdf)

- **FastFlowNet**
  - **Repo:** `https://github.com/ltkong218/FastFlowNet`
  - **Paper:** [Arxiv 2021](https://arxiv.org/pdf/2103.04524.pdf)

- **RAFT**
  - **Repo:** `https://github.com/princeton-vl/RAFT`
  - *Context:* Often used for flow initialization hooks.

- **DIS (Dense Inverse Search)**
  - **Paper:** [Arxiv 2016](https://arxiv.org/abs/1603.03590)

### Motion Modelling
- **AdaCoF (Adaptive Collaboration of Flows)**
  - **Repo:** `https://github.com/HyeongminLEE/AdaCoF-pytorch`
  - **Paper:** [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Lee_AdaCoF_Adaptive_Collaboration_of_Flows_for_Video_Frame_Interpolation_CVPR_2020_paper.html)

- **BMBC (Bilateral Motion + Cost Volume)**
  - **Repo:** `https://github.com/JunHeum/BMBC`

- **EBME (Enhanced Bi-directional Motion Estimation)**
  - **Repo:** `https://github.com/MCG-NKU/EBME` (Inferred/Search)
  - *Context:* Lightweight estimator integrating Softmax Splatting.

## 3. Implementation Plumbing (Real-Time Pipeline)

### TensorRT & Quantization
- **TensorRT Docs:** `https://developer.nvidia.com/tensorrt`
- **Working with Quantized Types:** `https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html`
- **Torch-TensorRT:** `https://pytorch.org/TensorRT/`
- **Practical Implementations:**
  - **enhancr:** `https://github.com/mafiosnik777/enhancr`
  - **ComfyUI-Rife-TensorRT:** `https://github.com/yuvraj108c/ComfyUI-Rife-Tensorrt`
  - **Unity-TensorRT:** `https://github.com/aman-tiwari/Unity-TensorRT`
- **PMQ-VE (Progressive Multi-Frame Quantization)**
  - *Context:* Backtracking-based calibration for INT8 VFI.
- **ST-MFNet Mini**
  - *Context:* Knowledge Distillation-Driven interpolation.
- **Knowledge Distillation (Distilling the Knowledge in a Neural Network)**
  - **Paper:** [Arxiv 2015](https://arxiv.org/abs/1503.02531)
  - *Context:* Seminal paper for Teacher-Student compression strategies.

### Zero-Copy & Graphics Interop
- **DXGI Desktop Duplication API:** `https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/desktop-dup-api`
  - **Sample:** `https://github.com/microsoft/Windows-classic-samples/blob/main/Samples/DXGIDesktopDuplication/README.md`
- **CUDA Interop Samples:** `https://github.com/NVIDIA/cuda-samples`
- **Vulkan External Memory:** `https://docs.vulkan.org/guide/latest/extensions/external.html`

## 4. Warping & Splatting Mathematics
- **Softmax Splatting**
  - **Repo:** `https://github.com/sniklaus/softmax-splatting`
  - **Paper:** [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Niklaus_Softmax_Splatting_for_Video_Frame_Interpolation_CVPR_2020_paper.pdf)

- **Super SloMo**
  - **Paper:** [CVPR 2018](https://openaccess.thecvf.com/content_CVPR_2018/papers/Jiang_Super_SloMo_High_CVPR_2018_paper.pdf)

- **PatchEX (High-Quality Real-Time Temporal Supersampling)**
  - *Context:* Evaluates forward vs backward warping for disocclusions.

- **SportsSloMo**
  - *Context:* Human-centric VFI with auxiliary losses.

- **Guided Frame Interpolation with Softmax Splatting**
  - *Status:* `[MISSING REPO URL]`

## 5. Older / General References
*(Source: 'Theoretical High-Performance Non-Neural Motion Estimation Pipeline.pdf')*
*Note: These references (2012-2016) focus on variational methods or older patch-match techniques.*

- **Zabih & Woodfill (1994): Non-parametric local transforms for computing visual correspondence**
  - *Context:* The original Census Transform paper.
- **Stein (2004): Efficient computation of optical flow using the census transform**
  - *Context:* Ternary Census variant.
- **Chambolle & Pock (2011): A First-Order Primal-Dual Algorithm for Convex Problems**
  - *Context:* General primal-dual scheme used in variational methods.
- **Bao et al. (2014): Fast edge-preserving PatchMatch for large displacement optical flow**
  - *Context:* PatchMatch-based optical flow optimization.
- **Besse et al. (2014): PMBP: PatchMatch Belief Propagation for Correspondence Field Estimation**
  - *Context:* PatchMatch Belief Propagation.
- **Efficient Coarse-To-Fine PatchMatch for Large Displacement Optical Flow**
  - **Link:** `https://openaccess.thecvf.com/content_cvpr_2016/papers/Hu_Efficient_Coarse-To-Fine_PatchMatch_CVPR_2016_paper.pdf`
- **Flow Fields: Dense Correspondence Fields**
  - **Link:** `https://openaccess.thecvf.com/content_iccv_2015/papers/Bailer_Flow_Fields_Dense_ICCV_2015_paper.pdf`
- **An Evaluation of Data Costs for Optical Flow**
  - *Context:* GCPR 2013 paper.
- **A Duality Based Approach for Realtime TV-L1 Optical Flow**
  - **Link:** `https://link.springer.com/chapter/10.1007/978-3-540-74936-3_22`
- **Fast Global Image Smoothing based on Weighted Least Squares**
  - *Context:* TIP 2014 paper.
- **A TV-L1 Optical Flow Method with Occlusion Detection**
  - *Context:* DAGM 2012 paper.
