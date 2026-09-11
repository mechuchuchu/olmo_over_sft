# Experimental Plan: SFT Overtraining and Label Smoothing

## 1. Research Question

본 실험의 목적은 Supervised Fine-Tuning (SFT)에서 동일한 데이터가 반복적으로 사용될 때 발생하는 overtraining 현상을 정량적으로 분석하고, label smoothing이 이러한 현상을 완화할 수 있는지를 검증하는 것이다.

구체적으로 다음의 가설을 검증한다.

1. SFT epoch가 증가할수록 training loss는 지속적으로 감소하지만 validation performance는 일정 시점 이후 포화 또는 악화될 것이다.
2. Epoch 증가에 따라 모델의 predictive entropy가 감소하고, pretrained base model의 output distribution으로부터 점점 멀어질 것이다.
3. 이러한 현상은 label smoothing을 적용함으로써 완화될 것이다.
4. Label smoothing은 단순히 최종 validation loss를 개선하는 것뿐 아니라, SFT 과정에서 발생하는 과도한 confidence 증가와 entropy 감소 자체를 완화할 것이다.

---

## 2. Dataset

실험에는 `allenai/Dolci-Instruct-SFT`를 사용한다.

Dolci-Instruct-SFT는 Olmo 3 7B Instruct SFT에 사용된 mixture로, 총 2,152,112개의 training samples로 구성되어 있다.

공개된 tokenized version의 통계에 따르면 전체 데이터는 약 1.706B tokens이며, 평균 sequence length는 약 793 tokens이다. Assistant response에 해당하는 trainable tokens는 약 789M tokens이다.

본 실험에서는 전체 dataset을 사용하는 대신, SFT overtraining 효과를 명확하게 관찰하기 위해 subset을 구성한다.

### Dataset split

- Total sampled examples: 20,000
- Training: 18,000
- Validation: 2,000
- Sampling seed: 42

20,000개의 example은 Dolci-Instruct-SFT에서 random sampling한다.

중요하게, **1,000-token truncation 또는 1,000-token filtering은 적용하지 않는다.**

각 conversation은 가능한 한 원래 sequence length를 유지한다. 따라서 dataset의 자연스러운 length distribution을 보존한다.

단, GPU memory를 위한 별도의 maximum sequence length를 사용할 경우 해당 조건은 모든 실험에서 동일하게 유지한다.

---

## 3. Sampling Strategy

단순히 20,000개의 sample을 무작위 추출하는 것만으로 끝내지 않고, sampling 후 dataset statistics를 기록한다.

다음 통계를 보고한다.

- Number of examples
- Total tokens
- Trainable/assistant tokens
- Mean sequence length
- Median sequence length
- 90th percentile sequence length
- 99th percentile sequence length
- Maximum sequence length
- Source/domain distribution

특히 Dolci-Instruct-SFT는 여러 source dataset의 mixture이므로, random sampling 결과에서 특정 source가 과도하게 적거나 많아지는지를 확인한다.

Main experiment에서는 **natural mixture distribution을 유지한 random sampling**을 사용한다. 따라서 source별 균등 sampling은 적용하지 않는다.

---

## 4. Base Model

7B-class dense language model을 사용한다.

Dolci-Instruct-SFT를 실제로 사용한 base checkpoint인 `allenai/Olmo-3-1025-7B`를 사용하여 dataset과 model의 mismatch를 최소화한다. 이미 SFT가 적용된 `Olmo-3-7B-Instruct` checkpoint가 아니라, Dolci-Instruct-SFT의 SFT 결과가 시작된 base model을 사용한다.

Base model은 모든 실험에서 동일하게 유지한다.

---

## 5. SFT Epoch Sweep

동일한 training examples를 반복해서 사용하는 효과를 보기 위해 하나의 run을 8 epoch까지 학습하고, 다음 checkpoint를 비교한다.

| Experiment | Epoch |
|---|---:|
| E1 | 1 |
| E2 | 2 |
| E4 | 4 |
| E8 | 8 |

각 checkpoint는 동일한 training trajectory에서 저장한다. 따라서 epoch 조건별로 별도의 run을 만들지 않고, 동일한 18,000개의 training examples가 1, 2, 4, 8회 반복된 시점을 비교한다.

모든 run은 learning rate를 `3e-6`으로 고정하고 8 epoch까지 학습한다. 1–8 epoch에서 명확한 validation degradation이나 entropy collapse가 관찰되지 않을 경우 추가적으로 16 epoch를 수행한다.

중요한 것은 각 run에서 optimizer, learning rate, batch size, sequence length, random seed 및 data ordering을 동일하게 유지하는 것이다. Epoch 비교는 별도 run의 최종 checkpoint가 아니라 동일한 8-epoch run에서 저장한 checkpoint를 사용한다.

---

## 6. Label Smoothing

각 8-epoch run에서 다음 세 가지 label smoothing coefficient를 비교한다.

| Condition | Label smoothing |
|---|---:|
| LS-0 | 0.00 |
| LS-05 | 0.05 |
| LS-10 | 0.10 |

따라서 main experiment는 label smoothing 조건별 3개 run과 각 run의 4개 checkpoint 조건으로 구성된다.

**3 label-smoothing conditions × 4 checkpoints = 12 checkpoint conditions**

Label smoothing은 training 시작부터 적용한다.

즉, 이미 overtraining이 발생한 checkpoint에 사후적으로 smoothing을 적용하는 것이 아니라, **SFT 전체 과정에서 label smoothing이 overconfidence/entropy collapse를 예방하는지**를 검증한다.

---

## 7. Controlled Training Variables

모든 실험에서 다음 조건을 고정한다.

- Base model
- Training dataset
- Validation dataset
- Data ordering
- Batch size
- Gradient accumulation
- Learning rate: `3e-6`
- Learning-rate scheduler: 없음(constant)
- Warmup ratio: `0`
- Optimizer
- Weight decay
- Gradient clipping
- Maximum sequence length
- Chat template
- Precision
- Random seed

특히 learning rate를 epoch별로 재조정하지 않으며, 8 epoch 전체에서 `3e-6`을 유지한다.

이는 epoch 자체를 독립변수로 만들기 위함이다.

---

## 8. Evaluation Metrics

각 run에서 1, 2, 4, 8 epoch checkpoint를 저장하여 동일한 training trajectory의 SFT 변화를 분석한다.

### 8.1 Training loss

Label smoothing이 적용된 training objective와 hard-label NLL을 분리해서 기록한다.

Smoothed training objective는 실제 학습에 사용되는 손실이다.

\[
L_{\mathrm{train}}^{\mathrm{smooth}}
\]

모델의 예측 confidence를 동일한 기준으로 비교하기 위해, label smoothing을 적용하지 않은 hard-label NLL도 별도로 계산한다.

\[
L_{\mathrm{train}}^{\mathrm{hard}}
=
-\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(y_i\mid x_i)
\]

`LS=0`에서는 두 값이 동일하고, `LS>0`에서는 두 값을 모두 보고한다. Epoch가 증가함에 따라 두 training loss가 어떻게 변화하는지 측정한다.

### 8.2 Validation hard NLL

\[
L_{\mathrm{val}}^{\mathrm{hard}}
\]

Validation에서는 모든 모델에 대해 smoothing을 적용하지 않은 hard-label NLL을 사용한다. 따라서 label smoothing 조건 사이에서도 동일한 기준으로 validation performance를 비교할 수 있다.

\[
\mathrm{Gap}
=
L_{\mathrm{val}}^{\mathrm{hard}}-L_{\mathrm{train}}^{\mathrm{hard}}
\]

이 hard-NLL gap을 이용하여 overfitting을 측정한다.

### 8.3 Token accuracy

Training 및 validation set에서 assistant target token accuracy를 측정한다.

---

## 9. Predictive Entropy

각 validation token에서 모델의 full-vocabulary predictive distribution을 얻어 entropy를 계산한다.

\[
H(p_\theta)
=
-\sum_{v}p_\theta(v)\log p_\theta(v)
\]

평균 predictive entropy를 epoch별로 비교한다.

핵심 가설은 다음과 같다.

\[
\text{Epoch}\uparrow
\Rightarrow
H(p_\theta)\downarrow
\]

즉, SFT를 반복할수록 모델이 특정 target에 대해 지나치게 확신하게 되는지를 확인한다.

Label smoothing을 적용한 경우 이 entropy 감소가 완화되는지를 비교한다.

---

## 10. Base Model과의 Distributional Shift

SFT 이전 base model과 SFT checkpoint가 동일한 validation token에 대해 생성하는 probability distribution을 비교한다.

가능하다면 다음 metric을 측정한다.

\[
D_{KL}
\left(
p_{\mathrm{SFT}}
\Vert
p_{\mathrm{Base}}
\right)
\]

Epoch 증가에 따른 KL divergence 증가 여부를 분석한다.

이를 통해 단순히 "SFT model이 더 잘 맞는다"는 것과 "base model의 distribution에서 과도하게 멀어진다"는 현상을 구분한다.

---

## 11. Calibration

모델 confidence의 변화를 분석하기 위해 다음 metric을 사용한다.

- Expected Calibration Error (ECE)
- Hard-label Negative Log-Likelihood (NLL)
- Brier score
- Reliability diagram

특히 token-level confidence와 token accuracy를 함께 분석한다.

예상되는 현상은 다음과 같다.

\[
\text{Epoch}\uparrow
\Rightarrow
\text{Confidence}\uparrow
\]

그러나 validation accuracy가 그만큼 증가하지 않는다면,

\[
\text{Confidence}
>
\text{Actual correctness}
\]

가 되어 calibration이 악화된다.

---

## 12. Main Analysis

가장 중요한 분석은 다음 네 가지 변수의 trajectory를 비교하는 것이다.

\[
\boxed{
\text{Epoch}
\rightarrow
\begin{cases}
\text{Train hard NLL}\\
\text{Validation hard NLL}\\
\text{Entropy}\\
D_{KL}(\text{SFT}\Vert\text{Base})
\end{cases}
}
\]

Label smoothing이 효과가 있다면 다음과 같은 차이를 기대한다.

### No label smoothing

\[
\text{Epoch}\uparrow
\Rightarrow
\begin{cases}
L_{\mathrm{train}}^{\mathrm{hard}}\downarrow\\
H(p)\downarrow\\
KL(\mathrm{SFT}\Vert\mathrm{Base})\uparrow\\
ECE\uparrow
\end{cases}
\]

### With label smoothing

\[
\text{Epoch}\uparrow
\Rightarrow
\begin{cases}
L_{\mathrm{train}}^{\mathrm{hard}}\downarrow\\
H(p)\downarrow\text{ more slowly}\\
KL\uparrow\text{ more slowly}\\
ECE\uparrow\text{ less}
\end{cases}
\]

즉, label smoothing이 **training performance 자체를 제한하는 regularizer**인지, 아니면 **overconfidence와 distributional shift를 선택적으로 억제하는 regularizer**인지 구분한다.

---

## 13. Additional Dataset-Size Experiment

Main experiment에서 효과가 확인되면 dataset size에 따른 scaling experiment를 추가한다.

세 가지 training set size를 사용한다.

| Dataset | Train examples |
|---|---:|
| Small | 5k |
| Main | 18k |
| Large | 100k |

각각에서 동일한 epoch sweep을 적용한다.

\[
1,2,4,8\ \text{epochs}
\]

이를 통해 다음 질문을 추가적으로 검증한다.

> 동일한 number of epochs가 아니라, **동일한 data repetition intensity가 overtraining의 핵심 원인인가?**

예를 들어 5k × 8 epochs와 100k × 8 epochs에서 entropy 및 KL trajectory가 크게 다르다면, overtraining은 단순한 epoch 수보다 **unique training data 대비 repeated exposure**와 더 밀접하게 관련될 수 있다.

---

## 14. Experimental Matrix

### Main experiment

| Run | Dataset | LS | Checkpoints |
|---|---:|---:|---|
| LS-0 | 18k | 0.00 | 1 / 2 / 4 / 8 epoch |
| LS-05 | 18k | 0.05 | 1 / 2 / 4 / 8 epoch |
| LS-10 | 18k | 0.10 | 1 / 2 / 4 / 8 epoch |

**총 3 runs, 12 checkpoint conditions**

### Scaling experiment

Main experiment에서 유의미한 효과가 확인된 후:

- 5k
- 18k
- 100k

에 대해 epoch 1/2/4/8을 비교한다.

---

## 15. Expected Contribution

본 실험은 단순히 "SFT를 오래 하면 overfitting된다"는 것을 확인하는 데 목적이 있지 않다.

핵심 질문은 다음과 같다.

> **SFT overtraining이 발생할 때 모델 내부의 predictive distribution은 어떻게 변화하며, label smoothing이 그 distributional change를 예방할 수 있는가?**

이를 위해 smoothed training objective, hard-label training NLL, validation hard NLL뿐 아니라

**entropy + confidence + calibration + base-model KL divergence**

를 함께 측정한다.

이렇게 하면 SFT overtraining을 단순한 generalization gap이 아니라 **predictive distribution의 변화**라는 관점에서 분석할 수 있다.

또한 dataset size scaling을 통해 이 현상이 단순히 epoch 수의 함수인지, 아니면 **동일한 training examples에 대한 반복 노출(repeated exposure)**의 함수인지 추가적으로 검증한다.

## 16. Practical Starting Point

첫 실험에서는 세 가지 label smoothing 조건을 각각 8 epoch까지 학습하고, 각 run에서 1/2/4/8 epoch checkpoint를 평가한다.

1. 18k / 8 epoch / LS=0
2. 18k / 8 epoch / LS=0.05
3. 18k / 8 epoch / LS=0.10

각 checkpoint에서 smoothed training objective, hard-label training NLL, validation hard NLL, entropy 및 KL divergence를 함께 기록한다.

참고로 Ai2의 공개 SFT recipe에서는 Olmo 3 7B에 전체 Dolci-Instruct-SFT를 1 epoch 학습했으며, 8×H100에서 약 9.5시간의 training time이 보고되어 있다. 공개 recipe에서는 한 epoch가 약 1,723 steps이고, 해당 설정에서 평균 sequence length가 약 840 tokens로 보고된다.

따라서 본 연구의 subset/epoch-sweep은 Olmo 3의 공식 recipe를 재현하려는 것이 아니라, **SFT repetition에 따른 distributional change를 controlled setting에서 분리하여 관찰하는 실험**으로 정의한다.
