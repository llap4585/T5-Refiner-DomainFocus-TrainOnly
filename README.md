# T5-Refiner-DomainFocus-TrainOnly
# 经过T5-Refiner-DomainFocus预处理后的数据微调训练代码

![Views](https://komarev.com/ghpvc/?username=llap4585&repo=T5-Refiner-DomainFocus-TrainOnly&label=Project%20Views&color=blue&style=flat-square)

> It is recommended that the data be preprocessed using the following project:  
> [T5-Refiner-DomainFocus](https://github.com/llap4585/T5-Refiner-DomainFocus)
>
> 数据建议经过以下项目预处理：
> [T5-Refiner-DomainFocus](https://github.com/llap4585/T5-Refiner-DomainFocus)
>
> データは、以下のプロジェクトを使用して前処理することをおすすめします（機械翻訳）：  
> [T5-Refiner-DomainFocus](https://github.com/llap4585/T5-Refiner-DomainFocus)

---
<a name="Introduction"></a>
## Introduction
[⭐️English](#english) | [⭐️中文](#chinese)


[日本語](#japanese) | [Deutsch](#deutsch) | [Français](#francais) | [Español](#espanol) | [हिन्दी](#hindi) | [한국어](#korean) | [Português](#portuguese)

### Introduction to Other Languages 

— **one-time *quick* machine translation only**, provided according to the version as of February 1, 2026:

Arabic العربية, Bengali বাংলা, Russian русский, Italian italiano, Dutch Nederlands, Swedish svenska

[Introduction to Other Languages](./Introduction-to-Other-Languages.md)

---

[Demo](#Demo) 

[Prerequisite - without experience using T5 or mT5](#Prerequisites)

[Requirements](#Requirements)

[References](#References)

[Privacy](#Privacy)

---
<a name="Demo"></a>
## 📡 Demo
**Due to copyright and privacy constraints associated with real clinical documents and academic literature used in testing, data is not directly displayed in this project.**

**由于测试所使用的真实临床文档与学术文献涉及版权与隐私问题，本项目未直接展示样例数据。**

**実際の臨床文書および学術文献は、著作権およびプライバシーの問題を含むため、本プロジェクトではサンプルデータを直接公開していません。**


### 📊 Evaluation
> Without adjusting the training strategy, the model may stop training prematurely, achieving only around 60% restoration accuracy.
> 
> More than half of the remaining 40% fails to reach semantically coherent results.

Based on preliminary testing with the mT5-base standard model:
* **Standard Model Performance**: The restoration rate for specialized terminology is estimated to be below 60%. The remaining 40% of results are often logically incoherent and unacceptable for professional use.
* **With DomainFocus Improvement**: The estimated restoration rate reaches 85%. Of the remaining 15% error margin, most are semantic synonyms, which greatly improves the overall readability and logical consistency of the text.

### 📊效果评估
> 如果不对训练策略进行调整，模型可能会在早期阶段提前停止训练，最终只能达到约 60% 的还原率。
> 
> 在剩余的 40% 结果中，超过一半无法达到语义通顺的效果。

根据初步测试对比，在 mT5-base 标准模型中：
* **标准模型表现**：在专业领域的词汇还原率估算在 60% 以下，剩余 40% 的还原结果逻辑混乱，几乎无法被业务接受。
* **本项目改进后**：专业词汇还原率估算达到了 85%。剩下的 15% 误差中，大部分是语义相近的词汇替代，极大地提高了文本的整体可读性和逻辑连贯性。

### 📊効果評価（機械翻訳）
> 学習戦略を調整しない場合、モデルが早期に学習を停止してしまい、復元率は約 60% にとどまる可能性があります。
> 
> 残りの 40% のうち、半数以上は意味的に自然な結果に達しません。

mT5-base標準モデルを用いた初期テストの比較：
* **標準モデルのパフォーマンス**：専門分野の語彙復元率は推定60%以下。残りの40%は論理が混乱しており、業務利用はほぼ不可能です。
* **本プロジェクトによる改善後**：専門語彙の復元率は推定85%に達しました。残りの15%の誤差の大部分は意味の近い語彙への置換であり、テキスト全体の可読性と論理的な一貫性が大幅に向上しました。
  
---

[Introduction](#Introduction)

---
## Prerequisites - without experience using T5 or mT5

**If you have experience in T5 or mT5**: [Requirements](#Requirements)

[google-research/multilingual-t5](https://github.com/google-research/multilingual-t5)

> **English:**  
> This project provides basic training code for T5/mT5 models.  
> Training focuses on fine-tuning a pretrained model to recover masked or corrupted text.  
> If you are new to T5 training, it is helpful to understand the T5 model, masking, and tokenization.
>
> **中文：**  
> 本项目提供用于训练 T5/mT5 模型的基础代码。  
> 训练内容主要是对已有的预训练模型进行微调，使其学会从被遮蔽或破损的文本中还原完整内容。  
> 如果你不熟悉 T5 训练流程，了解 T5 模型、Masking 和 Tokenization 即可。
>
> **日本語（機械翻訳）:**  
> 本プロジェクトは、T5/mT5 モデルを学習させるための基本的なトレーニングコードです。  
> 事前学習済みモデルをファインチューニングし、マスクされたテキストの復元を学習します。  
> T5モデル、Masking、Tokenization の基礎を理解していれば十分です。
>
[Introduction](#Introduction)

---
<a name="Requirements"></a>
## 🛠️ Requirements


```text
            
                
```

> **Equipment List:**
> 
> GPU: NVIDIA RTX 3060 Laptop GPU (6GB)
> 
> Memory: 64GB DDR4 (upgraded prior to the price increase😄😆)
> 
>Notice:
>
>All essential instructions are included as comments within the code.
>
>No separate Quickstart guide is provided.
>
>I hate Quickstart!

[Introduction](#Introduction)

---
<a name="References"></a>
## 💪References / Citation
```markdown
This project builds upon the T5 or mT5. If you use mT5, please cite:

@inproceedings{xue-etal-2021-mt5,
    title = "m{T}5: A Massively Multilingual Pre-trained Text-to-Text Transformer",
    author = "Xue, Linting  and
      Constant, Noah  and
      Roberts, Adam  and
      Kale, Mihir  and
      Al-Rfou, Rami  and
      Siddhant, Aditya  and
      Barua, Aditya  and
      Raffel, Colin",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.41",
    doi = "10.18653/v1/2021.naacl-main.41",
    pages = "483--498"
}

If you use this project, please cite it as:

@misc{llap4585,
    title={{T5-Refiner-DomainFocus-TrainOnly}: fine-tuning T5/mT5 models on data preprocessed by T5-Refiner-DomainFocus.},
    author={llap4585},
    howpublished = {\url{https://github.com/llap4585/T5-Refiner-DomainFocus-TrainOnly}},
    year={2026}
}

```

[Introduction](#Introduction)

---

<a name="Privacy"></a>
## 🛡️ Privacy & Security

**Local Processing Only:** This tool performs all operations locally on your machine. No medical reports, patient data, or sensitive information are uploaded to any external servers or cloud services. Your data remains under your control at all times.

**Third-party Disclaimer:** All third-party libraries required for operation are provided by the user's environment. These dependencies and their components are not under the management or control of this project.

**仅限本地处理：** 本工具的所有操作均在您的本地计算机上执行。不会将任何医疗报告、患者数据或敏感信息上传到任何外部服务器或云服务。您的数据始终由您掌控。

**第三方库声明：** 本工具运行所依赖的所有第三方库均由用户环境提供，这些第三方库及其相关组件不在本项目的管理与控制范围内。

[Introduction](#Introduction)

---
> **⚠️Disclaimer:** The non-English and non-Chinese versions of this documentation are provided for convenience only and were generated using machine translation. README may have been revised multiple times, and non-Chinese content may be missing. In case of any discrepancy, the Chinese version shall prevail. 


