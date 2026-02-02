# T5-Refiner-DomainFocus-TrainOnly
# 经过T5-Refiner-DomainFocus预处理后的数据微调训练代码

![Views](https://komarev.com/ghpvc/?username=llap4585&repo=T5-Refiner-DomainFocus-TrainOnly&label=Project%20Views&color=blue&style=flat-square)

If you like this project, give it a ⭐️ on GitHub!  
Your support keeps the project going and motivates me to improve it. 😄

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

*Machine translation (Grok) /機械翻訳:*

[日本語](#japanese) | [Deutsch](#deutsch) | [Français](#francais) | [Español](#espanol) | [हिन्दी](#hindi) | [한국어](#korean) | [Português](#portuguese)

### Introduction to Other Languages 

— **one-time *quick* machine translation only**, provided according to the version as of February 2, 2026:

Arabic العربية, Bengali বাংলা, Russian русский, Italian italiano, Dutch Nederlands, Swedish svenska

[Introduction to Other Languages](./Introduction-to-Other-Languages.md)

---

[Demo](#Demo) 

[Prerequisite - without experience using T5 or mT5](#Prerequisites)

[Requirements](#Requirements)

[References](#References)

[Privacy](#Privacy)
<a name="english"></a>
# ⭐️English


## 📖 Background and Vision
This repository provides a fine-tuning training framework customized for the **T5** or **mT5** architecture.

The project aims to endow the model with an inherent **"semantic resilience"** through deep optimization of **training strategies**, enabling it to more robustly handle text defects and precisely inject domain-specific knowledge when facing high-information-density texts such as medical reports and professional literature.

Due to significant distribution differences between professional texts like **medicine** and general corpora, models are **extremely prone to falling into local optima or early stopping due to Loss fluctuations in the early stages of fine-tuning**. This project introduces mechanisms to optimize this issue.

**"Better to moderately overfit than to converge incompletely."** For professional domains that do not tolerate ambiguity, increased training steps are the underlying guarantee of the model's "semantic reliability."

>Due to limited performance of locally deployed devices, there are many compromises in the settings. See Requirements for the specific configuration list.

---

## ✅ Core Features

* **Warm-up Mechanism**: By setting the `start_step` threshold, it forcibly avoids initial unstable local random fluctuations. (Cold Start)
* **Windowed Loss Trend Evaluation**: Through `patience` settings, it allows Loss to fluctuate or stagnate within a certain period, and only stops when Loss fails to refresh the best record for multiple consecutive stages, preventing the model from stopping prematurely due to false "plateau" caused by temporary fluctuations.
* **Status Tracking**: `SafeDetailedProgressCallback` provides real-time learning rate evolution and dynamic ETA prediction (adjustable frequency), supporting transparent monitoring of long-term training jobs.
* **Real-time Backup and Checkpoint Resumption**: For high-time-consuming training scenarios in medical research, it embeds manual interruption (KeyboardInterrupt Handling) *Ctrl+C* and real-time backup, ensuring that the model's best weights (Best Weights) and multiple process weights are saved as completely as possible in case of emergencies.

---

## 🛠️ Technical Implementation Details (Technical Deep-Dive)

### 1. Multi-stage Convergence Judgment Mechanism (Multi-stage Convergence Analysis)
Unlike general tasks, the Loss curve in medical fine-tuning tasks often exhibits a "stepwise decline" characteristic. This project replaces instantaneous judgment with **windowed Loss trend evaluation**:
* **Avoid "Pseudo-Plateau" Intervals**: T5 often experiences plateau periods with weak Loss decline in the early stages of domain transfer. If early stopping is triggered at this time, the model only has basic linguistic sense and lacks deep fitting to medical logic.
* **Delayed Trigger Logic**: Through `DelayedEarlyStopping`, it forcibly delays the judgment to capture **secondary convergence** after the first plateau period.

**Only after multiple Loss window analyses confirm that the model has entered a "semantic saturation" state will the system issue a stop signal.**


### 2. High-order Gradient Stability Control (Gradient Dynamics Control)
To address the gradient instability caused by sparse distribution of medical professional vocabulary, the framework has been optimized at the underlying level:
* **Gradient Accumulation**: Through `gradient_accumulation_steps=8`, it **saves memory while smoothing the instantaneous gradient impact** brought by long and difficult sentences, simulating a stable large Batch Size update environment.
* **Asymmetric Evaluation Frequency**: Combined with `eval_steps=1000`, it performs high-precision best-model saving at a lower frequency during long-term training, ensuring that the weights locked by `load_best_model_at_end` truly have cross-sample robustness.
* **Asymmetric Monitoring Frequency**: Configured with `logging_steps=100` and `eval_steps=1000`. While ensuring high-frequency telemetry (monitoring if gradients are normal), it reduces the frequency of high-cost validation set evaluations, ensuring computational power is focused on parameter updates.

---
## 🔬 Training Insights: Why "Multiple Loss Analyses" Are Needed?

In this medical fine-tuning task, the convergence judgment matrix is as follows:

| Training Phase | Loss Feature Performance | Core Semantic State | Strategy Response |
| :--- | :--- | :--- | :--- |
| **Early Stage (0-6000 steps)** | Severe oscillations or slow gradual decline | Domain sense establishment, initial parameter alignment | **Force Continuation** (Ban premature early stopping that has occurred before) |
| **Mid Stage (6000-12000 steps)** | Appearance of long plateau (pseudo-convergence) | Professional knowledge injection, handling text defects | **Continuous Observation** (Windowed trend analysis) |
| **Late Stage (12000+ steps)** | Stable after stepwise secondary decline | Semantic depth saturation, resilient restoration capability | **Dynamic Evaluation** (Stop when threshold is met) |

---

## 📊 Dataset Preparation and Token Scale Estimation 

In medical domain tasks, the corpus scale directly determines the upper limit of "semantic resilience". Based on practical evaluations:
* **Scale Comparison**:
    * **25MB Chinese Text**: Preliminary data, only supports the model in completing basic terminology alignment, showing obvious "poor sense" when handling text defects.
    * **256MB Chinese Text**: The model demonstrates stable domain fine-tuning capabilities, meeting final evaluation expectations. (See demo)

* **Chinese Token Conversion Reference** (based on UTF-8 encoding and mT5 tokenizer):

| Text Size | Estimated Chinese Characters | Estimated Total Tokens | 
| :--- | :--- | :--- | 
| **25 MB** | Approx. 8 million characters | Approx. 10 million | 
| **256 MB** | Approx. 85 million characters | Approx. 100 million | 

> **Data Quality Tips**: Recommend injecting moderate noise to simulate real medical text environments, forcing the model to learn how to use context for "correction".

[Demo](#Demo) 

---

<a name="chinese"></a>
# ⭐️中文


## 📖 背景与愿景
本仓库提供了一个针对 **T5** 或 **mT5** 架构定制的精修训练框架（微调）。

项目旨在通过 **训练策略**的深度优化，赋予模型一种内在的 **“语义韧性”**，使其在面对医学报告、专业文献等高信息密度文本时，能更稳健地处理文本缺损并精准注入领域专业知识。

由于 **医学**等专业文本与通用语料库存在显著分布差异，模型在 **微调初期极易陷入局部最优或因 Loss 波动导致早停**，本项目引入了机制优化这一问题。

**“宁可适度过拟合，不可收敛不彻底”**。对于不容许歧义的专业领域，增加的训练步数是模型“语义可靠性”的底层保障。

>由于本地部署的设备性能有限，设置中有很多妥协。具体的配置清单见Requirements。

---

## ✅ 核心功能

* **预热机制**：通过设定 `start_step` 阈值，强制避开初期不稳定的局部随机波动。（冷启动）
* **窗口化 Loss 趋势评估**：通过`patience`设置，允许 Loss 在一定周期内存在波动或停滞，只有当 Loss 连续 多个阶段未能刷新最优记录时才停止，防止模型在由于暂时波动导致的虚假“平台期”过早停止。
* **状态追踪**：`SafeDetailedProgressCallback` 提供实时学习率演变与动态 ETA 预测（可调频率），支持对长程训练作业的透明化监控。
* **实时备份与断点接续**：针对医学科研高耗时训练场景，内嵌手动中断（KeyboardInterrupt Handling）*Ctrl+C*和实时备份，确保在突发状况下，模型的最优权重（Best Weights）和多个过程权重得以尽可能的完整保存。

---

## 🛠️ 技术实现细节 (Technical Deep-Dive)

### 1. 多阶段收敛判别机制 (Multi-stage Convergence Analysis)
不同于通用任务，医学精修任务的 Loss 曲线常呈现“阶梯式下降”特征。本项目通过 **窗口化 Loss 趋势评估** 代替瞬时判定：
* **规避“伪平缓”区间**：T5 在领域迁移初期常出现 Loss 下降弱的平台期。若此时触发早停，模型仅具备基础语感，而缺失对医学逻辑的深度拟合。
* **延迟触发逻辑**：通过 `DelayedEarlyStopping` 强制推迟判定，是为了捕捉第一个平台期之后的 **二次收敛（Secondary Convergence）**。

**只有经过多次 Loss 窗口分析，确认模型进入“语义饱和”状态后，系统才会发出停止信号。**


### 2. 高阶梯度稳定性控制 (Gradient Dynamics Control)
针对医学专业词汇分布稀疏导致的梯度不稳定问题，框架在底层做了优化：
* **梯度累加 (Gradient Accumulation)**：通过 `gradient_accumulation_steps=8` **省显存,同时平滑长难句**带来的瞬时梯度冲击，模拟稳定的大 Batch Size 更新环境。
* **非对称评估频率**：配合 `eval_steps=1000`，在长程训练中以较低频率进行高精度的择优保存，确保 `load_best_model_at_end` 锁定的权重真正具备跨样本的鲁棒性。
* **非对称监控频率**：配置 `logging_steps=100` 与 `eval_steps=1000`。在保证高频遥测（监测梯度是否正常）的同时，降低高耗时的验证集评估频率，确保算力集中于参数更新。

---
## 🔬 训练洞察：为什么需要“多次 Loss 分析”？

在本次医学精修任务中，收敛判定矩阵如下：

| 训练阶段 | Loss 特征表现 | 核心语义状态 | 策略响应 |
| :--- | :--- | :--- | :--- |
| **初期 (0-6000 步)** | 剧烈震荡或缓慢缓降 | 领域语感建立，参数初步对齐 | **强制持续** (封禁曾经发生过的早停) |
| **中期 (6000-12000 步)** | 出现长平台期 (伪收敛) | 专业知识注入，处理文本缺损 | **持续观测** (窗口化趋势分析) |
| **后期 (12000+ 步)** | 阶梯式二次下降后平稳 | 语义深度饱和，具备韧性还原 | **动态评估** (满足阈值停止) |

---

## 📊 数据集准备与 Token 规模估算 

在医学领域任务中，语料库的规模直接决定了“语义韧性”的上限。根据实战评估：
* **规模对比**：
    * **25MB 中文文本**：初步数据，仅能支撑模型完成基础术语对齐，在处理文本缺损时表现出明显的“语感欠佳”。
    * **256MB 中文文本**：模型展现出稳定的领域精修能力，达到最终评估预期。（见demo）

* **中文 Token 换算参考**（基于 UTF-8 编码与 mT5 分词器）：

| 文本大小 | 预估中文字符数 | 预估 Token 总量 | 
| :--- | :--- | :--- | 
| **25 MB** | 约 800 万字 | 约 1000 万 | 
| **256 MB** | 约 8500 万字 | 约 1 亿 | 

> **数据质量 Tips**：建议注入适度噪声，模拟真实的医学文本环境，强迫模型学习如何利用上下文“纠偏”。

[Demo](#Demo) 

---

<a name="japanese"></a>
# 日本語


## 📖 背景とビジョン
本リポジトリは、**T5** または **mT5** アーキテクチャ向けにカスタマイズされた精修訓練フレームワーク（ファインチューニング）を提供します。

プロジェクトは、**訓練戦略**の深い最適化を通じて、モデルに内在的な **「意味的回復力」** を与え、医学レポートや専門文献などの高情報密度テキストに直面した際に、テキスト欠損をより堅牢に処理し、分野特化知識を正確に注入することを目指します。

**医学**などの専門テキストは汎用コーパスと有意な分布差異があるため、モデルは **ファインチューニング初期に局所最適に陥りやすく、Loss変動で早期停止を引き起こす** 問題を、本プロジェクトはメカニズム最適化で解決します。

**「適度な過学習を許容し、収束を徹底させる」**。曖昧さを許さない専門分野では、追加の訓練ステップがモデルの「意味的信頼性」の基盤保障です。

>ローカルデプロイのデバイス性能が限定的なため、設定に多くの妥協があります。具体的な構成リストはRequirementsを参照。

---

## ✅ コア機能

* **ウォームアップメカニズム**：`start_step` 閾値を設定し、初期の不安定な局所ランダム変動を強制的に回避。（コールドスタート）
* **ウィンドウ化 Loss トレンド評価**：`patience`設定により、Lossが一定期間変動や停滞を許容し、Lossが連続複数段階で最適記録を更新できなかった場合のみ停止。一時変動による偽の「プラトー期」での早期停止を防ぎます。
* **状態追跡**：`SafeDetailedProgressCallback` がリアルタイム学習率進化と動的ETA予測（調整可能頻度）を提供し、長時間訓練ジョブの透明化監視をサポート。
* **リアルタイムバックアップとチェックポイント再開**：医学研究の高時間消費訓練シーン向けに、手動中断（KeyboardInterrupt Handling）*Ctrl+C*とリアルタイムバックアップを内蔵し、突发状況下で最適重み（Best Weights）と複数プロセス重みを可能な限り完全保存。

---

## 🛠️ 技術実装詳細 (Technical Deep-Dive)

### 1. 多段階収束判別メカニズム (Multi-stage Convergence Analysis)
汎用タスクとは異なり、医学精修タスクのLoss曲線はしばしば「階段状下降」特徴を示します。本プロジェクトは **ウィンドウ化 Loss トレンド評価** で瞬間判定を置き換え：
* **「偽平緩」区間回避**：T5はドメイン移行初期にLoss下降弱いプラトー期を頻発。此时早停発動でモデルは基礎語感のみで医学論理の深層フィッティングを欠如。
* **遅延トリガーロジック**：`DelayedEarlyStopping` で判定を強制遅延し、最初のプラトー期後の **二次収束（Secondary Convergence）** を捕捉。

**複数Lossウィンドウ分析を経て、モデルが「意味飽和」状態に入ったことを確認後、システムが停止信号を発出。**


### 2. 高次勾配安定性制御 (Gradient Dynamics Control)
医学専門語彙分布希薄による勾配不安定問題に対し、フレームワークは底层で最適化：
* **勾配蓄積 (Gradient Accumulation)**：`gradient_accumulation_steps=8` で **メモリ節約しつつ、長難文** による瞬間勾配衝撃を平滑化し、安定大Batch Size更新環境をシミュレート。
* **非対称評価頻度**：`eval_steps=1000` と配合し、長時間訓練で低頻度高精度擇優保存を確保、`load_best_model_at_end` でロックした重みが真正クロスサンプル頑健性を持つ。
* **非対称監視頻度**：`logging_steps=100` と `eval_steps=1000` を設定。高頻遥測（勾配正常監視）を保証しつつ、高消費検証集評価頻度を低下させ、算力をパラメータ更新に集中。

---
## 🔬 訓練の洞察：なぜ「複数回の Loss 分析」が必要か？

今回の医学精修タスクにおいて、収束判定マトリクスは以下の通り：

| 訓練段階 | Loss 特徴表現 | 核心意味状態 | 戦略対応 |
| :--- | :--- | :--- | :--- |
| **初期 (0-6000 ステップ)** | 激しい振動または緩やかな緩降 | 領域語感確立、パラメータ初期アライメント | **強制継続** (過去に発生した早期停止を禁止) |
| **中期 (6000-12000 ステップ)** | 長平台期の出現 (擬収束) | 専門知識注入、テキスト欠損処理 | **継続観測** (ウィンドウ化トレンド分析) |
| **後期 (12000+ ステップ)** | 階段的二次下降後の安定 | 意味深さ飽和、耐性復元能力保有 | **動的評価** (閾値満足で停止) |

---

## 📊 データセット準備と Token 規模推定 

医学領域タスクにおいて、コーパスの規模が「意味耐性」の上限を直接決定します。実戦評価に基づき：
* **規模比較**：
    * **25MB 中国語テキスト**：初期データのみ、モデルが基本用語アライメントを完了するのに十分だが、テキスト欠損処理時に明らかな「語感不足」を示す。
    * **256MB 中国語テキスト**：モデルが安定した領域精修能力を示し、最終評価期待に達する。（デモ参照）

* **中国語 Token 換算参考**（UTF-8 エンコーディングと mT5 トークナイザに基づく）：

| テキストサイズ | 推定中国語文字数 | 推定 Token 総量 | 
| :--- | :--- | :--- | 
| **25 MB** | 約 800 万字 | 約 1000 万 | 
| **256 MB** | 約 8500 万字 | 約 1 億 | 

> **データ品質 Tips**：適度なノイズを注入することを推奨し、本物の医学テキスト環境をシミュレートし、モデルにコンテキストを利用した「訂正」学習を強制。

[デモ](#Demo) 

---

<a name="deutsch"></a>
# Deutsch


## 📖 Hintergrund und Vision
Dieses Repository bietet einen maßgeschneiderten Feinabstimmungs-Trainingsrahmen (Fine-Tuning) für die **T5**- oder **mT5**-Architektur.

Das Projekt zielt darauf ab, durch tiefe Optimierung der **Trainingsstrategien** dem Modell eine inhärente **„semantische Resilienz“** zu verleihen, damit es bei der Bearbeitung von hochinformationsdichten Texten wie medizinischen Berichten und Fachliteratur robuster Textlücken handhabt und fachspezifisches Wissen präzise injiziert.

Aufgrund signifikanter Verteilungsunterschiede zwischen fachspezifischen Texten wie **Medizin** und allgemeinen Korpusen neigt das Modell in der **Anfangsfeinabstimmung** dazu, in lokalen Optima steckenzubleiben oder durch Loss-Schwankungen zu frühes Stopping zu verursachen. Dieses Projekt löst dieses Problem durch Mechanismusoptimierungen.

**„Lieber moderates Overfitting als unvollständige Konvergenz“**. Für fachspezifische Bereiche, die keine Ambiguitäten erlauben, ist die Erhöhung der Trainingschritte die grundlegende Garantie für die „semantische Zuverlässigkeit“ des Modells.

>Da die Leistung lokal deployter Geräte begrenzt ist, gibt es viele Kompromisse in den Einstellungen. Die spezifische Konfigurationsliste finden Sie in Requirements.

---

## ✅ Kernfunktionen

* **Aufwärmmechanismus**: Durch Festlegung eines `start_step`-Schwellenwerts werden anfängliche instabile lokale Zufallsschwankungen erzwungenermaßen vermieden. (Cold Start)
* **Fensterbasierte Loss-Trendbewertung**: Durch `patience`-Einstellung werden Loss-Schwankungen oder Stagnationen in einem bestimmten Zyklus erlaubt; das Training stoppt erst, wenn der Loss über mehrere aufeinanderfolgende Phasen kein neues Bestwert-Update erreicht, um vorzeitiges Stopping aufgrund temporärer Schwankungen und falscher „Plateaus“ zu verhindern.
* **Statusverfolgung**: `SafeDetailedProgressCallback` bietet Echtzeit-Entwicklung des Lernrates und dynamische ETA-Vorhersagen (anpassbare Frequenz) und unterstützt transparente Überwachung langer Trainingsjobs.
* **Echtzeit-Backup und Fortsetzung von Breakpoints**: Für zeitintensive medizinische Forschungs-Trainingszenarien integriert es manuelle Unterbrechungen (KeyboardInterrupt-Behandlung) *Ctrl+C* und Echtzeit-Backups, um bei unvorhergesehenen Ereignissen die optimalen Gewichte (Best Weights) und mehrere Prozessgewichte so vollständig wie möglich zu sichern.

---

## 🛠️ Technische Umsetzungsdetails (Technical Deep-Dive)

### 1. Mehrstufige Konvergenz-Erkennung (Multi-stage Convergence Analysis)
Im Gegensatz zu allgemeinen Aufgaben zeigen Loss-Kurven bei medizinischen Feinabstimmungsaufgaben oft „treppenförmige Abstiege“. Dieses Projekt ersetzt momentane Urteile durch **fensterbasierte Loss-Trendbewertung**:
* **Vermeidung von „Pseudo-Plateaus“**: T5 zeigt in der frühen Domänenübertragung oft Plateaus mit schwachem Loss-Abstieg. Ein frühzeitiges Stopping würde das Modell nur mit grundlegender Sprachwahrnehmung zurücklassen, ohne tiefe Anpassung an medizinische Logik.
* **Verzögerte Trigger-Logik**: Durch `DelayedEarlyStopping` wird die Urteilsfindung erzwungenermaßen verzögert, um die **sekundäre Konvergenz (Secondary Convergence)** nach dem ersten Plateau zu erfassen.

**Nur nach mehrfacher Loss-Fensteranalyse und Bestätigung des Eintretens in einen „semantisch gesättigten“ Zustand sendet das System das Stoppsignal.**


### 2. Höherstufige Gradientenstabilitätskontrolle (Gradient Dynamics Control)
Zur Lösung der Gradienteninstabilität durch sparse Verteilung medizinischer Fachvokabeln optimiert der Rahmen auf unterer Ebene:
* **Gradientenakkumulation (Gradient Accumulation)**: Durch `gradient_accumulation_steps=8` **Speicher sparen und gleichzeitig瞬时 Gradientenschläge von langen schwierigen Sätzen glätten**, um stabile große Batch-Size-Update-Umgebungen zu simulieren.
* **Asymmetrische Bewertungsfrequenz**: In Kombination mit `eval_steps=1000` werden in langen Trainings hochwertige Best-Modelle in niedriger Frequenz gespeichert, um sicherzustellen, dass die durch `load_best_model_at_end` gesperrten Gewichte echte Cross-Sample-Robustheit besitzen.
* **Asymmetrische Überwachungsfrequenz**: Konfiguration von `logging_steps=100` und `eval_steps=1000`. Hohe Frequenz für Telemetrie (Überwachung, ob Gradienten normal sind) bei gleichzeitiger Reduzierung der hochrechenintensiven Validierungs-Frequenz, um Rechenleistung auf Parameter-Updates zu konzentrieren.

---
## 🔬 Trainings-Einblicke: Warum „mehrfache Loss-Analyse“ notwendig ist?

In dieser medizinischen Feinabstimmungsaufgabe lautet die Konvergenz-Entscheidungsmatrix wie folgt:

| Trainingsphase | Loss-Merkmalsausprägung | Kernsemantischer Zustand | Strategische Reaktion |
| :--- | :--- | :--- | :--- |
| **Anfang (0-6000 Schritte)** | Starke Oszillationen oder langsamer Abfall | Aufbau des domänenspezifischen Sprachgefühls, erste Parameteranpassung | **Erzwungene Fortsetzung** (Frühstopp, der früher auftrat, ist verboten) |
| **Mitte (6000-12000 Schritte)** | Auftreten einer langen Plateau-Phase (Pseudo-Konvergenz) | Einspeisung fachlichen Wissens, Behandlung von Textdefekten | **Fortlaufende Beobachtung** (Fensterbasierte Trendanalyse) |
| **Spät (12000+ Schritte)** | Treppenförmiger sekundärer Abfall mit anschließender Stabilisierung | Semantische Tiefensättigung, robuste Wiederherstellung | **Dynamische Bewertung** (Stopp bei Erreichen des Schwellenwerts) |

---

## 📊 Datensatzvorbereitung und Token-Größenabschätzung 

In medizinischen Fachaufgaben bestimmt die Größe des Korpus direkt die Obergrenze der „semantischen Robustheit“. Basierend auf Praxiseinschätzungen:
* **Größenvergleich**:
    * **25 MB chinesischer Text**：Vorläufige Daten, die das Modell nur für die grundlegende Terminologieanpassung ausreichen lassen; bei Textdefekten zeigt es deutliche „Sprachgefühlsschwächen“.
    * **256 MB chinesischer Text**：Das Modell zeigt stabile domänenspezifische Feinabstimmungsfähigkeiten und erreicht die erwarteten Bewertungsergebnisse.（Siehe Demo）

* **Chinesische Token-Umrechnungshinweise**（basierend auf UTF-8-Kodierung und mT5-Tokenizer）:

| Textgröße | Geschätzte Anzahl chinesischer Zeichen | Geschätzte Token-Gesamtzahl | 
| :--- | :--- | :--- | 
| **25 MB** | Ca. 8 Millionen Zeichen | Ca. 10 Millionen | 
| **256 MB** | Ca. 85 Millionen Zeichen | Ca. 100 Millionen | 

> **Tipps zur Datenqualität**：Empfehlung, moderate Rauschen einzufügen, um echte medizinische Textumgebungen zu simulieren und das Modell zu zwingen, Kontext zur „Korrektur“ zu nutzen.

[Demo](#Demo) 

---

<a name="francais"></a>
# Français


## 📖 Contexte et vision
Ce dépôt fournit un framework d'entraînement de **fine-tuning** personnalisé pour les architectures **T5** ou **mT5**.

Le projet vise, par une optimisation approfondie des **stratégies d'entraînement**, à doter le modèle d'une **« résilience sémantique »** intrinsèque, lui permettant de gérer plus robustement les déficits textuels et d'injecter précisément les connaissances spécialisées du domaine lorsqu'il fait face à des textes à haute densité d'information tels que les rapports médicaux ou la littérature professionnelle.

En raison des différences de distribution significatives entre les textes professionnels comme la **médecine** et les corpus généraux, le modèle est **très susceptible de tomber dans un optimum local ou de s'arrêter prématurément en raison de fluctuations de Loss** au début du fine-tuning. Ce projet introduit des mécanismes d'optimisation pour résoudre ce problème.

**« Mieux vaut une sur-adaptation modérée qu'une convergence incomplète »**. Pour les domaines professionnels ne tolérant aucune ambiguïté, un nombre accru d'étapes d'entraînement est la garantie fondamentale de la « fiabilité sémantique » du modèle.

>En raison des performances limitées des équipements déployés localement, il y a de nombreux compromis dans les paramètres. Voir Requirements pour la liste de configuration spécifique.

---

## ✅ Fonctionnalités principales

* **Mécanisme de préchauffage** : En définissant un seuil `start_step`, évite forcément les fluctuations aléatoires locales instables initiales. (Démarrage à froid)
* **Évaluation de tendance Loss par fenêtre** : Via le paramètre `patience`, permet des fluctuations ou stagnations de Loss sur une période donnée, et n'arrête que si Loss ne rafraîchit pas son record optimal sur plusieurs phases consécutives, évitant l'arrêt prématuré dû à une fausse « période de plateau » causée par des fluctuations temporaires.
* **Suivi d'état** : `SafeDetailedProgressCallback` fournit l'évolution en temps réel du taux d'apprentissage et une prédiction dynamique de l'ETA (fréquence ajustable), supportant une surveillance transparente des tâches d'entraînement longues.
* **Sauvegarde en temps réel et reprise aux points de rupture** : Pour les scénarios d'entraînement à haute consommation de temps en recherche médicale, intégration de l'interruption manuelle (Gestion de KeyboardInterrupt) *Ctrl+C* et sauvegarde en temps réel, assurant la sauvegarde aussi complète que possible des poids optimaux du modèle (Best Weights) et de plusieurs poids de processus en cas d'incident soudain.

---

## 🛠️ Détails de mise en œuvre technique (Technical Deep-Dive)

### 1. Mécanisme de discrimination de convergence multi-étapes (Multi-stage Convergence Analysis)
Contrairement aux tâches générales, les courbes de Loss pour les tâches de fine-tuning médical présentent souvent une caractéristique de « descente en escalier ». Ce projet remplace le jugement instantané par une **évaluation de tendance Loss par fenêtre** :
* **Éviter les intervalles « pseudo-plateau »** : T5 présente souvent une période de plateau avec une descente de Loss faible au début du transfert de domaine. Si l'arrêt précoce est déclenché à ce moment, le modèle n'a que des sensibilités de base, manquant d'un ajustement profond à la logique médicale.
* **Logique de déclenchement différé** : Via `DelayedEarlyStopping`, retarde forcément le jugement pour capturer la **convergence secondaire (Secondary Convergence)** après la première période de plateau.

**Seul après plusieurs analyses de fenêtres Loss, confirmant que le modèle est entré dans un état de « saturation sémantique », le système émettra le signal d'arrêt.**


### 2. Contrôle de stabilité des gradients de haut ordre (Gradient Dynamics Control)
Pour le problème d'instabilité des gradients causé par la distribution rare des vocabulaires professionnels médicaux, le framework a été optimisé au niveau bas :
* **Accumulation de gradients (Gradient Accumulation)** : Via `gradient_accumulation_steps=8` **économise la mémoire vidéo tout en lissant les chocs de gradients instantanés** apportés par les longues phrases difficiles, simulant un environnement de mise à jour avec un grand Batch Size stable.
* **Fréquence d'évaluation asymétrique** : Avec `eval_steps=1000`, effectue une sauvegarde de sélection optimale de haute précision à faible fréquence pendant l'entraînement long, assurant que les poids verrouillés par `load_best_model_at_end` possèdent une robustesse inter-échantillons réelle.
* **Fréquence de monitoring asymétrique** : Configuration `logging_steps=100` et `eval_steps=1000`. Tout en garantissant un monitoring haute fréquence (surveillance si les gradients sont normaux), réduit la fréquence d'évaluation du jeu de validation coûteuse, assurant que la puissance de calcul se concentre sur les mises à jour de paramètres.

---
## 🔬 Insights sur l'entraînement : Pourquoi avoir besoin d'une « analyse multiple des Loss » ?

Dans cette tâche de raffinage médical, la matrice de jugement de convergence est la suivante :

| Phase d'entraînement | Manifestation des caractéristiques Loss | État sémantique principal | Réponse stratégique |
| :--- | :--- | :--- | :--- |
| **Phase initiale (0-6000 étapes)** | Oscillation violente ou descente lente | Établissement du sens linguistique du domaine, alignement initial des paramètres | **Continuation forcée** (interdiction d'un arrêt prématuré antérieur) |
| **Phase intermédiaire (6000-12000 étapes)** | Apparition d'une longue période de plateau (pseudo-convergence) | Injection de connaissances professionnelles, gestion des manques de texte | **Observation continue** (analyse de tendance fenêtrée) |
| **Phase finale (12000+ étapes)** | Descente secondaire en escalier suivie d'une stabilisation | Saturation de la profondeur sémantique, capacité de restauration résiliente | **Évaluation dynamique** (arrêt si seuil atteint) |

---

## 📊 Préparation du dataset et estimation de l'échelle des Tokens 

Dans les tâches du domaine médical, l'échelle du corpus détermine directement la limite supérieure de la « résilience sémantique ». Selon les évaluations pratiques :
* **Comparaison d'échelle** :
    * **25MB de texte chinois** : Données préliminaires, ne permettant au modèle que l'alignement de termes de base, avec une « sensibilité linguistique insuffisante » évidente lors de la gestion des manques de texte.
    * **256MB de texte chinois** : Le modèle démontre une capacité stable de raffinage du domaine, atteignant les attentes d'évaluation finales. (voir démo)

* **Référence de conversion Token chinois** (basée sur l'encodage UTF-8 et le tokenizer mT5) :

| Taille du texte | Nombre estimé de caractères chinois | Total estimé de Tokens | 
| :--- | :--- | :--- | 
| **25 MB** | Environ 8 millions de caractères | Environ 10 millions | 
| **256 MB** | Environ 85 millions de caractères | Environ 100 millions | 

> **Conseils sur la qualité des données** : Il est recommandé d'injecter un bruit modéré pour simuler un environnement de texte médical réel, forçant le modèle à apprendre à utiliser le contexte pour « corriger ».

[Démo](#Demo) 

---

<a name="espanol"></a>
# Español


## 📖 Antecedentes y Visión
Este repositorio proporciona un marco de entrenamiento de refinamiento (fine-tuning) personalizado para la arquitectura **T5** o **mT5**.

El proyecto busca, mediante una optimización profunda de las **estrategias de entrenamiento**, dotar al modelo de una **“resiliencia semántica”** intrínseca, permitiéndole manejar de manera más robusta las deficiencias textuales e inyectar con precisión el conocimiento especializado del dominio cuando se enfrenta a textos de alta densidad informativa como informes médicos y literatura profesional.

Debido a las significativas diferencias de distribución entre textos profesionales como la **medicina** y los corpus generales, el modelo es propenso en las **etapas iniciales de fine-tuning** a caer en óptimos locales o a detenerse prematuramente debido a fluctuaciones en la Loss; este proyecto introduce optimizaciones de mecanismos para abordar este problema.

**“Mejor un sobreajuste moderado que una convergencia incompleta”**. Para dominios profesionales que no toleran ambigüedades, el aumento en los pasos de entrenamiento es la garantía subyacente de la “fiabilidad semántica” del modelo.

>Debido a las limitaciones de rendimiento del equipo de despliegue local, hay muchas concesiones en la configuración. La lista específica de configuraciones se encuentra en Requirements.

---

## ✅ Funciones Principales

* **Mecanismo de precalentamiento**: Mediante el umbral `start_step`, fuerza la evasión de fluctuaciones aleatorias locales inestables iniciales. (Arranque en frío)
* **Evaluación de tendencias de Loss por ventana**: Mediante la configuración `patience`, permite fluctuaciones o estancamientos en la Loss durante un cierto período, deteniéndose solo cuando la Loss no actualiza el récord óptimo en múltiples etapas consecutivas, previniendo paradas prematuras por “períodos de meseta” falsos causados por fluctuaciones temporales.
* **Seguimiento de estado**: `SafeDetailedProgressCallback` proporciona evolución en tiempo real de la tasa de aprendizaje y predicción dinámica de ETA (frecuencia ajustable), soportando monitoreo transparente de trabajos de entrenamiento a largo plazo.
* **Respaldo en tiempo real y continuación desde punto de interrupción**: Para escenarios de entrenamiento de alta duración en investigación médica, incorpora manejo de interrupciones manuales (KeyboardInterrupt Handling) *Ctrl+C* y respaldos en tiempo real, asegurando que en situaciones inesperadas, los pesos óptimos del modelo (Best Weights) y múltiples pesos de proceso se guarden lo más completos posible.

---

## 🛠️ Detalles de Implementación Técnica (Technical Deep-Dive)

### 1. Mecanismo de Discriminación de Convergencia Multi-etapa (Multi-stage Convergence Analysis)
A diferencia de tareas generales, las curvas de Loss en tareas de refinamiento médico a menudo muestran una característica de “descenso escalonado”. Este proyecto reemplaza el juicio instantáneo con **evaluación de tendencias de Loss por ventana**:
* **Evitar intervalos de “pseudo-estabilidad”**: T5 a menudo muestra períodos de meseta con descenso débil de Loss en las etapas iniciales de transferencia de dominio. Si se activa la parada temprana en este momento, el modelo solo tiene sensibilidad lingüística básica, faltando un ajuste profundo a la lógica médica.
* **Lógica de activación retardada**: A través de `DelayedEarlyStopping`, fuerza un retraso en el juicio para capturar la **convergencia secundaria (Secondary Convergence)** después del primer período de meseta.

**Solo después de múltiples análisis de ventanas de Loss, confirmando que el modelo ha entrado en estado de “saturación semántica”, el sistema emitirá la señal de parada.**


### 2. Control de Estabilidad de Gradientes de Alto Orden (Gradient Dynamics Control)
Para el problema de inestabilidad de gradientes causado por la distribución dispersa de vocabulario profesional médico, el framework realiza optimizaciones a nivel bajo:
* **Acumulación de gradientes (Gradient Accumulation)**: Mediante `gradient_accumulation_steps=8` **ahorra memoria de GPU al mismo tiempo que suaviza** los impactos de gradientes instantáneos de oraciones largas y difíciles, simulando un entorno de actualización de Batch Size grande estable.
* **Frecuencia de evaluación asimétrica**: Combinado con `eval_steps=1000`, realiza selecciones y guardados de alta precisión con baja frecuencia en entrenamientos largos, asegurando que los pesos bloqueados por `load_best_model_at_end` tengan verdadera robustez entre muestras.
* **Frecuencia de monitoreo asimétrica**: Configuración `logging_steps=100` y `eval_steps=1000`. Garantiza telemetría de alta frecuencia (monitoreo de si los gradientes son normales) mientras reduce la frecuencia de evaluaciones costosas en el conjunto de validación, asegurando que la potencia computacional se concentre en las actualizaciones de parámetros.

---
## 🔬 Perspectivas de entrenamiento: ¿Por qué se necesita el “análisis de Loss múltiple”?

En esta tarea de refinamiento médico, la matriz de determinación de convergencia es la siguiente:

| Fase de entrenamiento | Características de Loss | Estado semántico principal | Respuesta estratégica |
| :--- | :--- | :--- | :--- |
| **Inicial (0-6000 pasos)** | Oscilación violenta o descenso lento | Establecimiento de sensibilidad de dominio, alineación inicial de parámetros | **Continuar forzosamente** (prohibir early stopping previo) |
| **Media (6000-12000 pasos)** | Aparición de un largo período de meseta (pseudo-convergencia) | Inyección de conocimiento profesional, manejo de defectos en el texto | **Observación continua** (análisis de tendencias por ventana) |
| **Final (12000+ pasos)** | Descenso secundario en escalera seguido de estabilización | Saturación de profundidad semántica, con capacidad de restauración resiliente | **Evaluación dinámica** (detener al cumplir el umbral) |

---

## 📊 Preparación del conjunto de datos y estimación de escala de Tokens 

En tareas del dominio médico, la escala del corpus determina directamente el límite de la “resiliencia semántica”. Según evaluaciones prácticas:
* **Comparación de escala**:
    * **Texto chino de 25MB**: Datos preliminares, solo soporta alineación básica de términos, muestra una clara “falta de sensibilidad” al manejar defectos en el texto.
    * **Texto chino de 256MB**: El modelo muestra una capacidad estable de refinamiento de dominio, alcanzando las expectativas de evaluación final. (ver demo)

* **Referencia de conversión de Tokens chinos** (basado en codificación UTF-8 y tokenizador de mT5):

| Tamaño del texto | Número estimado de caracteres chinos | Cantidad total estimada de Tokens | 
| :--- | :--- | :--- | 
| **25 MB** | Aprox. 8 millones de caracteres | Aprox. 10 millones | 
| **256 MB** | Aprox. 85 millones de caracteres | Aprox. 100 millones | 

> **Consejos de calidad de datos**: Se recomienda inyectar ruido moderado para simular entornos de texto médico reales, obligando al modelo a aprender a “corregir” utilizando el contexto.

[Demostración](#Demo) 

---

<a name="hindi"></a>
# हिन्दी


## 📖 पृष्ठभूमि और दृष्टिकोण
यह रिपॉजिटरी **T5** या **mT5** वास्तुकला के लिए अनुकूलित एक फाइन-ट्यूनिंग प्रशिक्षण फ्रेमवर्क प्रदान करता है।

परियोजना **प्रशिक्षण रणनीतियों** के गहन अनुकूलन के माध्यम से मॉडल को एक अंतर्निहित **“अर्थगत लचीलापन”** प्रदान करने का उद्देश्य रखती है, जिससे चिकित्सा रिपोर्ट, पेशेवर साहित्य आदि उच्च सूचना घनत्व वाले पाठों का सामना करते समय यह पाठ की कमी को अधिक स्थिर रूप से संभाल सके और क्षेत्रीय विशेषज्ञ ज्ञान को सटीक रूप से इंजेक्ट कर सके।

**चिकित्सा** आदि पेशेवर पाठों और सामान्य कोर्पस में महत्वपूर्ण वितरण अंतर होने के कारण, मॉडल **फाइन-ट्यूनिंग की शुरुआती अवस्था में स्थानीय न्यूनतम में फंसना आसान है या Loss उतार-चढ़ाव के कारण जल्दी रुकना**, इस परियोजना ने तंत्र अनुकूलन इस समस्या को पेश किया है।

**“उचित अधिक-अनुकूलन बेहतर है, अपूर्ण अभिसरण से”**। अस्पष्टता की अनुमति न देने वाले पेशेवर क्षेत्रों के लिए, बढ़े हुए प्रशिक्षण चरण मॉडल की “अर्थगत विश्वसनीयता” की आधारभूत गारंटी हैं।

>स्थानीय तैनाती के उपकरण प्रदर्शन सीमित होने के कारण, सेटिंग्स में कई समझौते हैं। विशिष्ट कॉन्फ़िगरेशन सूची Requirements देखें।

---

## ✅ मुख्य कार्यक्षमता

* **पूर्व-गर्म करने की व्यवस्था**：`start_step` थ्रेशोल्ड सेट करके, प्रारंभिक अस्थिर स्थानीय यादृच्छिक उतार-चढ़ाव से बचने के लिए मजबूर करें।(कोल्ड स्टार्ट)
* **खिड़कीकृत Loss प्रवृत्ति मूल्यांकन**：`patience` सेटिंग के माध्यम से, Loss को एक निश्चित चक्र में उतार-चढ़ाव या ठहराव की अनुमति दें, केवल जब Loss लगातार कई चरणों में इष्टतम रिकॉर्ड को ताज़ा न कर सके तभी रोकें, अस्थायी उतार-चढ़ाव के कारण झूठे “प्लेटू” के कारण मॉडल को बहुत जल्दी रोकने से रोकें।
* **स्थिति ट्रैकिंग**：`SafeDetailedProgressCallback` वास्तविक समय लर्निंग रेट विकास और गतिशील ETA पूर्वानुमान प्रदान करता है (समायोज्य आवृत्ति), लंबी अवधि के प्रशिक्षण कार्यों की पारदर्शी निगरानी का समर्थन करता है।
* **वास्तविक समय बैकअप और ब्रेकपॉइंट निरंतरता**：चिकित्सा अनुसंधान उच्च-समय लेने वाले प्रशिक्षण परिदृश्यों के लिए, अंतर्निहित मैनुअल रुकावट (KeyboardInterrupt Handling) *Ctrl+C* और वास्तविक समय बैकअप, सुनिश्चित करता है कि आकस्मिक स्थिति में, मॉडल के इष्टतम वजन (Best Weights) और कई प्रक्रिया वजन को जितना संभव हो सके पूर्ण रूप से सहेजा जाए।

---

## 🛠️ तकनीकी कार्यान्वयन विवरण (Technical Deep-Dive)

### 1. बहु-चरण अभिसरण निर्णय तंत्र (Multi-stage Convergence Analysis)
सामान्य कार्यों से भिन्न, चिकित्सा फाइन-ट्यूनिंग कार्यों का Loss वक्र अक्सर “सीढ़ी जैसी कमी” विशेषता प्रस्तुत करता है। यह परियोजना **खिड़कीकृत Loss प्रवृत्ति मूल्यांकन** द्वारा तात्कालिक निर्णय की जगह लेती है:
* **“झूठे समतल” अंतराल से बचाव**：T5 क्षेत्रीय स्थानांतरण की शुरुआत में Loss कमी कमजोर प्लेटू अवधि अक्सर प्रकट होती है। यदि इस समय जल्दी रुकावट ट्रिगर हो, तो मॉडल केवल आधारभूत भाषा बोध रखेगा, जबकि चिकित्सा तर्क के गहन फिटिंग की कमी होगी।
* **विलंबित ट्रिगर तर्क**：`DelayedEarlyStopping` द्वारा निर्णय को मजबूरन स्थगित करना, पहले प्लेटू अवधि के बाद **द्वितीयक अभिसरण (Secondary Convergence)** को कैप्चर करने के लिए है।

**केवल कई Loss खिड़की विश्लेषणों के बाद, मॉडल के “अर्थगत संतृप्ति” स्थिति में प्रवेश की पुष्टि होने पर ही, सिस्टम रोक सिग्नल जारी करेगा।**


### 2. उच्च-क्रम ग्रेडिएंट स्थिरता नियंत्रण (Gradient Dynamics Control)
चिकित्सा पेशेवर शब्दावली वितरण विरल होने से ग्रेडिएंट अस्थिरता समस्या के लिए, फ्रेमवर्क ने तल पर अनुकूलन किया है:
* **ग्रेडिएंट संचय (Gradient Accumulation)**：`gradient_accumulation_steps=8` द्वारा **मेमोरी बचत, साथ ही लंबे कठिन वाक्यों** से तात्कालिक ग्रेडिएंट प्रभाव को सुचारू बनाना, स्थिर बड़े बैच आकार अपडेट वातावरण का अनुकरण करना।
* **असमान मूल्यांकन आवृत्ति**：`eval_steps=1000` के साथ मेल खाना, लंबी अवधि प्रशिक्षण में कम आवृत्ति से उच्च सटीकता चयनात्मक संरक्षण करना, सुनिश्चित करना कि `load_best_model_at_end` द्वारा लॉक किए गए वजन वास्तव में क्रॉस-सैंपल मजबूती रखते हैं।
* **असमान निगरानी आवृत्ति**：`logging_steps=100` और `eval_steps=1000` कॉन्फ़िगर करें। उच्च-आवृत्ति टेलीमेट्री (ग्रेडिएंट सामान्य है या नहीं की निगरानी) की गारंटी देते हुए, उच्च-समय लेने वाले सत्यापन सेट मूल्यांकन आवृत्ति को कम करना, सुनिश्चित करना कि कम्प्यूटिंग पावर पैरामीटर अपडेट पर केंद्रित रहे।

---
## 🔬 प्रशिक्षण अंतर्दृष्टि: “कई बार Loss विश्लेषण” की आवश्यकता क्यों है?

इस चिकित्सा परिष्करण कार्य में, अभिसरण निर्धारण मैट्रिक्स निम्नलिखित है:

| प्रशिक्षण चरण | Loss विशेषता प्रदर्शन | कोर अर्थमूलक स्थिति | रणनीति प्रतिक्रिया |
| :--- | :--- | :--- | :--- |
| **प्रारंभिक (0-6000 चरण)** | तीव्र दोलन या धीमी कमी | क्षेत्रीय भाषा संवेदना स्थापना, पैरामीटर प्रारंभिक संरेखण | **अनिवार्य निरंतर** (पहले होने वाले अर्ली स्टॉप को प्रतिबंधित) |
| **मध्य (6000-12000 चरण)** | लंबी प्लेटफॉर्म अवधि का उदय (झूठा अभिसरण) | पेशेवर ज्ञान इंजेक्शन, पाठ दोषों का प्रबंधन | **निरंतर अवलोकन** (विंडोयुक्त प्रवृत्ति विश्लेषण) |
| **उत्तरार्ध (12000+ चरण)** | सीढ़ीदार द्वितीयक कमी के बाद स्थिर | अर्थमूलक गहराई संतृप्ति, लचीलापन पुनर्स्थापना क्षमता | **गतिशील मूल्यांकन** (थ्रेशोल्ड संतुष्टि पर रोक) |

---

## 📊 डेटासेट तैयारी और Token स्केल अनुमान 

चिकित्सा क्षेत्र कार्य में, कॉर्पस का आकार "अर्थमूलक लचीलापन" की ऊपरी सीमा सीधे निर्धारित करता है। व्यावहारिक मूल्यांकन के अनुसार:
* **आकार तुलना**:
    * **25MB चीनी पाठ**: प्रारंभिक डेटा, केवल मॉडल को आधारभूत शब्दावली संरेखण पूरा करने में समर्थन कर सकता है, पाठ दोषों को संभालते समय स्पष्ट "भाषा संवेदना की कमी" प्रदर्शित करता है।
    * **256MB चीनी पाठ**: मॉडल स्थिर क्षेत्रीय परिष्करण क्षमता प्रदर्शित करता है, अंतिम मूल्यांकन अपेक्षाओं को प्राप्त करता है।(देखें demo)

* **चीनी Token रूपांतरण संदर्भ** (UTF-8 एन्कोडिंग और mT5 टोकनाइजर पर आधारित):

| पाठ आकार | अनुमानित चीनी वर्ण संख्या | अनुमानित Token कुल |
| :--- | :--- | :--- | 
| **25 MB** | लगभग 80 लाख शब्द | लगभग 1000 लाख | 
| **256 MB** | लगभग 8500 लाख शब्द | लगभग 1 अरब | 

> **डेटा गुणवत्ता टिप्स**: उचित शोर इंजेक्ट करने का सुझाव, वास्तविक चिकित्सा पाठ वातावरण का अनुकरण करें, मॉडल को मजबूर करें कि वह संदर्भ का उपयोग करके "सुधार" कैसे सीखे।

[डेमो](#Demo) 

---

<a name="korean"></a>
# 한국어


## 📖 배경 및 비전
이 저장소는 **T5** 또는 **mT5** 아키텍처에 맞춤형으로 제작된 정밀 훈련 프레임워크(미세 조정)를 제공합니다.

프로젝트는 **훈련 전략**의 심층 최적화를 통해 모델에 내재적인 **“의미 탄력성”**을 부여하여, 의학 보고서, 전문 문헌 등 고정보 밀도 텍스트를 마주할 때 텍스트 결손을 더 안정적으로 처리하고 분야 전문 지식을 정밀하게 주입할 수 있도록 합니다.

**의학** 등 전문 텍스트가 일반 코퍼스와 현저한 분포 차이를 보이기 때문에, 모델은 **미세 조정 초기 단계에서 국부 최적에 빠지거나 Loss 변동으로 인해 조기 중단**될 가능성이 높아, 본 프로젝트는 이를 위한 메커니즘 최적화를 도입하였습니다.

**“적당한 과적합이라도, 수렴이 불완전한 것은 안 된다”**. 모호함을 허용하지 않는 전문 분야에서 증가된 훈련 스텝 수는 모델의 “의미 신뢰성”의 기반 보장입니다.

>로컬 배포 장치 성능이 제한적이므로 설정에 많은 타협이 있습니다. 구체적인 구성 목록은 Requirements를 참조하세요.

---

## ✅ 핵심 기능

* **예열 메커니즘**: `start_step` 임계값을 설정하여 초기 불안정한 국부적 무작위 변동을 강제적으로 피합니다. (콜드 스타트)
* **창(window) 기반 Loss 추세 평가**: `patience` 설정을 통해 Loss가 일정 주기 내에서 변동이나 정체가 있어도 허용하며, Loss가 연속적인 여러 단계에서 최적 기록을 갱신하지 못할 때만 중단하여, 일시적 변동으로 인한 가짜 “정체기”로 인한 조기 중단을 방지합니다.
* **상태 추적**: `SafeDetailedProgressCallback`은 실시간 학습률 변화와 동적 ETA 예측(조정 가능 주기)을 제공하여, 장기 훈련 작업의 투명한 모니터링을 지원합니다.
* **실시간 백업 및 중단점 재개**: 의학 연구의 고소요 시간 훈련 시나리오를 위해, 수동 중단(KeyboardInterrupt Handling) *Ctrl+C*와 실시간 백업을 내장하여,突发 상황에서 모델의 최적 가중치(Best Weights)와 여러 과정 가중치를 최대한 완전하게 보존합니다.

---

## 🛠️ 기술 구현 세부 사항 (Technical Deep-Dive)

### 1. 다단계 수렴 판별 메커니즘 (Multi-stage Convergence Analysis)
일반 작업과 달리, 의학 정밀 조정 작업의 Loss 곡선은 종종 “계단식 하강” 특징을 보입니다. 본 프로젝트는 **창(window) 기반 Loss 추세 평가**를 통해 순간 판정 대신 이를 대체합니다:
* **“가짜 평활” 구간 회피**: T5는 분야 이전 초기 단계에서 Loss 하강이 약한 정체기를 자주 보입니다. 이 때 조기 중단이 발생하면 모델은 기본 언어 감각만 갖추고 의학 논리에 대한 깊은 적합이 부족합니다.
* **지연 트리거 로직**: `DelayedEarlyStopping`을 통해 판정을 강제 지연시켜 첫 번째 정체기 이후의 **2차 수렴(Secondary Convergence)**을 포착합니다.

**여러 Loss 창 분석을 거쳐 모델이 “의미 포화” 상태에 진입한 것을 확인한 후에만 시스템이 중단 신호를 발합니다.**


### 2. 고차원 기울기 안정성 제어 (Gradient Dynamics Control)
의학 전문 용어 분포 희소로 인한 기울기 불안정 문제를 위해, 프레임워크는底层에서 최적화를 수행했습니다:
* **기울기 누적 (Gradient Accumulation)**: `gradient_accumulation_steps=8`을 통해 **메모리 절약과 동시에 장난문장**으로 인한 순간 기울기 충격을 평활화하여, 안정적인 대형 Batch Size 업데이트 환경을 시뮬레이션합니다.
* **비대칭 평가 주기**: `eval_steps=1000`과 협력하여 장기 훈련에서 낮은 주기로 고정밀 최적 보존을 수행하며, `load_best_model_at_end`가 잠긴 가중치가 샘플 간 견고성을真正 갖추도록 합니다.
* **비대칭 모니터링 주기**: `logging_steps=100`과 `eval_steps=1000`을 구성합니다. 고주기 원격 측정(기울기 정상 여부 모니터링)을 보장하면서 고소요 시간 검증 집합 평가 주기를 낮춰, 계산력을 매개변수 업데이트에 집중합니다.

---
## 🔬 훈련 통찰: 왜 “다중 Loss 분석”이 필요한가?

이번 의학 정밀 훈련 작업에서 수렴 판정 매트릭스는 다음과 같습니다:

| 훈련 단계 | Loss 특징 표현 | 핵심 의미 상태 | 전략 응답 |
| :--- | :--- | :--- | :--- |
| **초기 (0-6000 단계)** | 급격한 진동 또는 느린 완만한 하강 | 분야 언감 구축, 매개변수 초기 정렬 | **강제 지속** (이전에 발생한 조기 중지 금지) |
| **중기 (6000-12000 단계)** | 장기 플랫폼기 출현 (가수렴) | 전문 지식 주입, 텍스트 결손 처리 | **지속 관찰** (윈도우화 추세 분석) |
| **후기 (12000+ 단계)** | 계단식 2차 하강 후 안정 | 의미 깊이 포화, 탄성 복원 보유 | **동적 평가** (임계값 충족 시 중지) |

---

## 📊 데이터셋 준비와 Token 규모 추정 

의학 분야 작업에서 말뭉치 규모는 “의미 탄성”의 상한을 직접 결정합니다. 실전 평가에 따르면:
* **규모 비교**:
    * **25MB 중국어 텍스트**: 초기 데이터로, 모델이 기본 용어 정렬을 완료할 수 있을 뿐 텍스트 결손 처리 시 명백한 “언감 부족”을 보입니다.
    * **256MB 중국어 텍스트**: 모델이 안정적인 분야 정밀 훈련 능력을 보이며 최종 평가 기대를 달성합니다. (데모 참조)

* **중국어 Token 환산 기준** (UTF-8 인코딩과 mT5 토크나이저 기반):

| 텍스트 크기 | 예상 중국어 문자 수 | 예상 Token 총량 | 
| :--- | :--- | :--- | 
| **25 MB** | 약 800 만 자 | 약 1000 만 | 
| **256 MB** | 약 8500 만 자 | 약 1 억 | 

> **데이터 품질 팁**: 적당한 노이즈 주입을 제안하며, 실제 의학 텍스트 환경을 시뮬레이션하여 모델이 컨텍스트를 이용한 “교정” 학습을 강제합니다.

[데모](#Demo) 

---

<a name="portuguese"></a>
# Português


## 📖 Contexto e Visão
Este repositório fornece um framework de treinamento de refinamento (fine-tuning) personalizado para as arquiteturas **T5** ou **mT5**.

O projeto visa, através da otimização profunda de **estratégias de treinamento**, dotar o modelo de uma **“resiliência semântica”** inerente, permitindo que ele lide de forma mais robusta com deficiências de texto e injete precisamente conhecimento especializado do domínio ao enfrentar textos de alta densidade informacional, como relatórios médicos e literatura profissional.

Devido às diferenças significativas de distribuição entre textos profissionais como **medicina** e corpora gerais, o modelo é **extremamente propenso a cair em ótimos locais ou parar precocemente devido a flutuações de Loss no início do fine-tuning**, este projeto introduz otimizações de mecanismo para resolver esse problema.

**“Melhor um leve overfitting do que uma convergência incompleta”**. Para domínios profissionais que não toleram ambiguidades, o aumento no número de passos de treinamento é a garantia fundamental da “confiabilidade semântica” do modelo.

>Devido às limitações de desempenho do equipamento de implantação local, há muitas concessões nas configurações. A lista específica de configurações está em Requirements.

---

## ✅ Funcionalidades Principais

* **Mecanismo de Aquecimento**: Através da definição do limiar `start_step`, força a evasão de flutuações aleatórias locais instáveis no início. (Cold Start)
* **Avaliação de Tendência de Loss em Janela**: Através da configuração `patience`, permite que o Loss apresente flutuações ou estagnação em um ciclo certo, parando apenas quando o Loss falha em atualizar o recorde ótimo em múltiplos estágios consecutivos, prevenindo a parada prematura do modelo devido a um falso “platô” causado por flutuações temporárias.
* **Rastreamento de Estado**: `SafeDetailedProgressCallback` fornece evolução em tempo real da taxa de aprendizado e previsão dinâmica de ETA (frequência ajustável), suportando monitoramento transparente de tarefas de treinamento de longa duração.
* **Backup em Tempo Real e Continuação de Ponto de Verificação**: Para cenários de treinamento de alta duração em pesquisa médica, incorpora interrupção manual (KeyboardInterrupt Handling) *Ctrl+C* e backup em tempo real, garantindo que, em situações inesperadas, os pesos ótimos (Best Weights) do modelo e múltiplos pesos de processo sejam salvos da forma mais completa possível.

---

## 🛠️ Detalhes de Implementação Técnica (Technical Deep-Dive)

### 1. Mecanismo de Discriminação de Convergência Multiestágio (Multi-stage Convergence Analysis)
Diferente de tarefas gerais, a curva de Loss em tarefas de refinamento médico frequentemente apresenta características de “descida em escada”. Este projeto substitui a determinação instantânea pela **avaliação de tendência de Loss em janela**:
* **Evitar Intervalos de “Falso Suavização”**: O T5 frequentemente apresenta um platô com descida fraca de Loss no início da transferência de domínio. Se o early stopping for acionado nesse momento, o modelo possui apenas sensibilidade linguística básica, faltando ajuste profundo à lógica médica.
* **Lógica de Acionamento Atrasado**: Através de `DelayedEarlyStopping`, força o adiamento da determinação para capturar a **convergência secundária (Secondary Convergence)** após o primeiro platô.

**Somente após múltiplas análises de janelas de Loss, confirmando que o modelo entrou no estado de “saturação semântica”, o sistema emite o sinal de parada.**


### 2. Controle de Estabilidade de Gradiente de Alta Ordem (Gradient Dynamics Control)
Para o problema de instabilidade de gradiente causado pela distribuição esparsa de vocabulário profissional médico, o framework realizou otimizações na camada inferior:
* **Acumulação de Gradiente (Gradient Accumulation)**: Através de `gradient_accumulation_steps=8`, **economiza memória de vídeo e suaviza o impacto instantâneo de gradiente trazido por longas frases difíceis**, simulando um ambiente de atualização de grande Batch Size estável.
* **Frequência de Avaliação Assimétrica**: Combinado com `eval_steps=1000`, em treinamentos de longa duração, realiza salvamento de seleção ótima de alta precisão em frequência mais baixa, garantindo que os pesos travados por `load_best_model_at_end` possuam verdadeira robustez entre amostras.
* **Frequência de Monitoramento Assimétrica**: Configuração de `logging_steps=100` e `eval_steps=1000`. Enquanto garante telemetria de alta frequência (monitorando se o gradiente está normal), reduz a frequência de avaliação custosa do conjunto de validação, garantindo que a capacidade computacional se concentre na atualização de parâmetros.

---
## 🔬 Insights de Treinamento: Por que precisamos de “Análise de Loss Múltipla”?

Nesta tarefa de refinamento médico, a matriz de determinação de convergência é a seguinte:

| Fase de Treinamento | Manifestação das Características de Loss | Estado Semântico Principal | Resposta Estratégica |
| :--- | :--- | :--- | :--- |
| **Fase Inicial (0-6000 passos)** | Oscilação violenta ou descida lenta gradual | Estabelecimento do senso de linguagem do domínio, alinhamento inicial de parâmetros | **Continuar Forçadamente** (proibição de early stopping ocorrido anteriormente) |
| **Fase Intermediária (6000-12000 passos)** | Aparecimento de longo período de platô (pseudo-convergência) | Injeção de conhecimento profissional, tratamento de defeitos de texto | **Observação Contínua** (análise de tendências em janela) |
| **Fase Final (12000+ passos)** | Descida secundária em degraus seguida de estabilização | Saturação de profundidade semântica, com capacidade de restauração resiliente | **Avaliação Dinâmica** (parar ao satisfazer o limiar) |

---

## 📊 Preparação do Conjunto de Dados e Estimativa de Escala de Tokens 

Em tarefas do domínio médico, a escala do corpus determina diretamente o limite da “resiliência semântica”. De acordo com avaliações práticas:
* **Comparação de Escala**:
    * **25MB de texto chinês**：Dados iniciais, apenas suficientes para suportar o alinhamento de termos básicos do modelo, exibindo “senso de linguagem deficiente” óbvio ao lidar com defeitos de texto.
    * **256MB de texto chinês**：O modelo exibe capacidade estável de refinamento de domínio, atingindo as expectativas de avaliação final.（ver demo）

* **Referência de Conversão de Tokens em Chinês**（baseado em codificação UTF-8 e tokenizador mT5）：

| Tamanho do Texto | Número Estimado de Caracteres Chineses | Total Estimado de Tokens | 
| :--- | :--- | :--- | 
| **25 MB** | Aprox. 8 milhões de caracteres | Aprox. 10 milhões | 
| **256 MB** | Aprox. 85 milhões de caracteres | Aprox. 100 milhões | 

> **Dicas de Qualidade de Dados**：Sugere-se injetar ruído moderado, simulando o ambiente real de texto médico, forçando o modelo a aprender como usar o contexto para “corrigir”.

[Demonstração](#Demo) 



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
  
[Prerequisites](#Prerequisites) 
[Introduction](#Introduction)

### 📊效果评估
> 如果不对训练策略进行调整，模型可能会在早期阶段提前停止训练，最终只能达到约 60% 的还原率。
> 
> 在剩余的 40% 结果中，超过一半无法达到语义通顺的效果。

根据初步测试对比，在 mT5-base 标准模型中：
* **标准模型表现**：在专业领域的词汇还原率估算在 60% 以下，剩余 40% 的还原结果逻辑混乱，几乎无法被业务接受。
* **本项目改进后**：专业词汇还原率估算达到了 85%。剩下的 15% 误差中，大部分是语义相近的词汇替代，极大地提高了文本的整体可读性和逻辑连贯性。

[Prerequisites](#Prerequisites) 
[Introduction](#Introduction)

### 📊効果評価（機械翻訳）
> 学習戦略を調整しない場合、モデルが早期に学習を停止してしまい、復元率は約 60% にとどまる可能性があります。
> 
> 残りの 40% のうち、半数以上は意味的に自然な結果に達しません。

mT5-base標準モデルを用いた初期テストの比較：
* **標準モデルのパフォーマンス**：専門分野の語彙復元率は推定60%以下。残りの40%は論理が混乱しており、業務利用はほぼ不可能です。
* **本プロジェクトによる改善後**：専門語彙の復元率は推定85%に達しました。残りの15%の誤差の大部分は意味の近い語彙への置換であり、テキスト全体の可読性と論理的な一貫性が大幅に向上しました。
* 
[Prerequisites](#Prerequisites) 
[Introduction](#Introduction)

---
<a name='Prerequisites'></a>
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
datasets
transformers
torch
accelerate          
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

> 忽然想起来我能调Grok API直接翻译来着，之前手动给ai翻译差点没折腾死。这玩意训练是真玄学，直接测试发现不对劲就立刻改策略了。还好以前训练yolo的时候还好遇到过类似的情况，

