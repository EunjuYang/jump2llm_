# 🎛 Jump into LLM 


This is an `awesome` repository designed to guide individuals into the world of LLMs (Large Language Models). It is intended for those who already have some knowledge of deep learning, so it may not be suitable for complete beginners.

The repository contains a curated list of papers, websites, and videos to help deepen your understanding of the LLM field. Contributions and discussions are always welcome!

## Index

| **Step** | **Section**                                  | **Subsections**                                                  |
|----------|----------------------------------------------|------------------------------------------------------------------|
| 1        | [Starting Point](#starting-point)            | [Transformer](#transformer)                                     |
| 2        | [Understanding the Training Paradigm of LLMs](#understanding-the-training-paradigm-of-llms) | [Pre-Training & Fine-Tuning](#pre-training--fine-tuning) |
| 3        | [Understanding the Training Paradigm of LLMs](#understanding-the-training-paradigm-of-llms) | [Parameter-Efficient Fine-Tuning (PEFT)](#parameter-efficient-fine-tuning-peft) |
| 4        | [Understanding the Training Paradigm of LLMs](#understanding-the-training-paradigm-of-llms) | [In-Context Learning (ICL)](#in-context-learning-icl) |
| Appendix | [Understanding NLP Tasks](#understanding-nlp-tasks)              | |


## 🚩 Starting Point 

### 🤖 Transformer 
If you're new to exploring the LLM field, start by understanding the Transformer architecture, which is the foundational building block of most LLMs. Keep in mind that the **T** in GPT (a model you likely associate with LLMs) stands for Transformer!

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 📚
  - Transformer (Encoder-Decoder model)
  - Additional hepful materials
    - 📃 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
    - 📼 [What is GPT?](https://youtu.be/wjZofJX0v4M?feature=shared) : helpful for  understanding an abstract & having an intuition on Transformer.

## ⚙️ Undestanding the Training Paradigm of LLM

### Pre-Training & Fine-Tuning

Once you understand the Transformer architecture and its functionality, it's essential to grasp the key training paradigms. Understanding LLMs goes beyond just knowing the network architecture—it also involves learning how these models acquire knowledge and how that knowledge is integrated into the network.

When you start reading papers in the LLM field, you'll likely come across terms like pre-training and fine-tuning (as well as zero-shot, one-shot, etc.). Before diving deeper, it's important to understand these concepts, which were introduced in the original GPT paper. Remember, GPT stands for Generative **Pretrained** Transformer!

- [Improving Language Understanding by Generative Pre-Training](https://hayate-lab.com/wp-content/uploads/2023/05/43372bfa750340059ad87ac8e538c53b.pdf) 📚
  - GPT-1 paper!
  - Decoder-only model
  - Next Token Prediction-based Language Modeling
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
  - Encoder model
  - Another pre-training method is proposed.
  - Has stregth on Natural Language Understanding (NLU)

### Parameter Efficient Fine-Tuning (PEFT)

Pre-training involves acquiring knowledge from large corpus data, while fine-tuning focuses on adapting the model to a specific task using a corresponding dataset. However, fully fine-tuning all the parameters of a network (known as full fine-tuning) is resource-intensive. To address this, several approaches that fine-tune only a subset of parameters have been introduced. These are referred to as Parameter-Efficient Fine-Tuning (PEFT).

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) 📚
    - One of the popular PEFT methods.
- There are many PEFT schemes, e.g., prefix tuning, P-tuning, IA3, etc. One can easily adopt the methods by using [PEFT 🤗](https://huggingface.co/docs/peft/index) library!



### In-Context Learning (ICL)

The pre-training and fine-tuning paradigm supports building task-specific expert models. Following this approach, we need to fine-tune a model for each individual task.

However, a new paradigm has emerged that removes the boundaries between tasks, suggesting that pre-training alone is sufficient to handle multiple tasks without the need for fine-tuning. This approach, known as In-Context Learning (ICL), fully leverages the power of pre-training by eliminating the fine-tuning step. In ICL, task information—referred to as context—is provided as input, enabling the pre-trained model to adapt to specific tasks.

The concept of In-Context Learning (ICL) was introduced with GPT-3. However, it is also valuable to read both GPT-2 and GPT-3 to gain a comprehensive understanding of the evolution and capabilities of these models.

  - GPT-2: [Language Models are Unsupervised Multitask Learners](https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf) 📚
    - The authors showed the power of pre-training, encompassing multi-task!
  - GPT-3: [Language Models are Few-Shot Learners
](https://arxiv.org/abs/2005.14165) 📚
    - The authors illustrate how GPT-3 leverages pre-training through in-context learning, which is an early example of prompt engineering.
    - Be sure to understand the concepts of Zero-Shot, One-Shot, and Few-Shot learning.
 - Chain of Thought (CoT) : [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
    - This paper demonstrates how prompting can enhance reasoning abilities in LLMs.
    - It provides insights into how 'context' and 'prompting' can effectively improve an LLM’s performance without the need for fine-tuning.
  

## 🧩 Understanding NLP Tasks
 
To fully grasp LLMs, it’s important to have a foundational understanding of Natural Language Processing (NLP) tasks. Each task involves a pair of input and desired output, with specific objectives, benchmarks, and evaluation metrics.

| **NLP Task**        | **Type**                | **Benchmarks**                                                                                   |
|---------------------|-------------------------|-------------------------------------------------------------------------------------------------|
| Token Classification | Classification          | [Benchmarks](https://paperswithcode.com/task/token-classification)                           |
| Translation          | Seq2Seq                 | [Benchmarks](https://paperswithcode.com/task/machine-translation)                             |
| Summarization        | Seq2Seq                 | [Benchmarks](https://paperswithcode.com/task/text-summarization)                               |
| Question Answering   | Span Extraction / Extractive | [Benchmarks](https://paperswithcode.com/task/question-answering)                              |


- Please find more details explanation from [here](https://huggingface.co/learn/nlp-course/chapter7/1?fw=pt)!

## 🎊 Update Info

- `24.09.08`: The initial README has been updated.
    - The section on LLM models will be updated soon.
    - The section on understanding the internals of LLMs will be updated soon.
- This README currently includes a curated selection of papers for beginners. Additional sub-README files for each section are being prepared.
