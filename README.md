# SurgicalVLM-Agent: Towards an Interactive AI Co-Pilot for Pituitary Surgery

## Abstract: 
Image-guided surgery demands adaptive, real-time decision support, yet static AI models struggle with integrating surgical data and providing interactive guidance. Large vision-language models (VLMs) offer a promising solution by enabling dynamic task planning and predictive decision support. We introduce SurgicalVLM-Agent, an AI co-pilot for image-guided pituitary surgery, capable of planning, conversation, and task execution. The agent dynamically processes surgeon queries and plans the tasks such as instrument tracking, endoscope anatomy segmentation, MRI tumor segmentation, overlaying preoperative imaging with intraoperative views and surgical visual question answering (VQA). To enable structured task planning, we develop PitAgent-dataset, a surgical context-aware dataset covering phase recognition, instrument localization, tool tracking, segmentation, overlaying, tool-tissue interactions, and surgical activity recognition. Additionally, we propose FFT-GaLore, an FFT-based gradient projection technique for efficient low-rank adaptation, optimizing fine-tuning for LLaMA 3.2 in surgical environments. We validate SurgicalVLM-Agent by assessing task planning and prompt generation on our PitAgent dataset and evaluating zero-shot VQA using a public pituitary dataset. Results demonstrate state-of-the-art performance in task planning and query interpretation, with highly semantically meaningful VQA responses, advancing AI-driven surgical assistance.

##Training Command
```
python main.py
```

## Inference Command
```
python inference.py
```

## Acknowlwdgement
The implementation of SurgicalVLM-Agent relies on resources from <a href="https://github.com/jiaweizzhao/GaLore">GaLore</a>, <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, <a href="https://github.com/huggingface/peft">PEFT</a>. We thank the original authors for their open-sourcing.
