# SurgicalLLM-Agent: Towards an Interactive AI Co-Pilot for Pituitary Surgery

## Abstract: 
Image-guided surgery demands adaptive, real-time decision support, yet static AI models struggle with structured task planning and providing interactive guidance. Large language models (LLMs)-powered agents offer a promising solution by enabling dynamic task planning and predictive decision support. Despite recent advances, the absence of surgical agent datasets and robust parameter-efficient fine-tuning techniques limits the development of LLM agents capable of complex intraoperative reasoning. In this paper, we introduce Surgical AI Copilot, an LLM agent for image-guided pituitary surgery, capable of conversation, planning, and task execution in response to queries involving tasks such as MRI tumor segmentation, endoscope anatomy segmentation, overlaying preoperative imaging with intraoperative views, instrument tracking, and surgical visual question answering (VQA). To enable structured agent planning, we develop the PitAgent dataset, a surgical context-aware planning dataset covering surgical tasks like workflow analysis, instrument localization, anatomical segmentation, and query-based reasoning. Additionally, we propose DEFT-GaLore, a Deterministic Energy-based Fourier Transform (DEFT) gradient projection technique for efficient low-rank adaptation of recent LLMs (e.g., LLaMA 3.2, Qwen 2.5), enabling their use as surgical agent planners.  We extensively validate our agent's performance and the proposed adaptation technique against other state-of-the-art low-rank adaptation methods on agent planning and prompt generation tasks, including a zero-shot surgical VQA benchmark, demonstrating the significant potential for truly efficient and scalable surgical LLM agents in real-time operative settings.

## PitAgent Dataset:
The training and testing dataset is included in the supplementary_materials\Dataset folder

## SurgicalVLM-Agent Pretrained Weights
The pretrained weights will be released upon the acceptance of the paper

## Training Command
```
python main.py `
    --model_name "meta-llama/Llama-3.2-3B-Instruct" `
    --train_file "C:\path\to\your\train.csv" `
    --val_file "C:\path\to\your\val.csv" `
    --save_path "C:\path\to\save\model" `
    --seed 50 `
    --HF_TOKEN "hf_your_huggingface_access_token_here"
```

## Inference Command
```
python inference.py `
    --best_model_path "C:\path\to\your\trained\model" `
    --input_files "test1.csv,test2.csv,test3.csv" `
    --output_dir "path\to\your\inference_output" `
    --seed 50 `
    --HF_TOKEN "hf_your_huggingface_access_token_here"
```

## Acknowledgement
The implementation of Surgical LLM Agent relies on resources from <a href="https://github.com/jiaweizzhao/GaLore">GaLore</a>, <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, <a href="https://github.com/huggingface/peft">PEFT</a>. We thank the original authors for their open-sourcing.
