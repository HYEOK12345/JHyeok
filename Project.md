# CNN 기반 흑색종 분류 및 OpenCV 병변 분석 진단 보조 시스템

> **팀명**: 同舟共濟 ("같은 배를 탄 사람들이 서로 돕는다")  
> **진행 기간**: 2025년 4월 ~ 6월  
> **역할**: 모델링 및 성능 평가 담당 (사진혁)

---

## 1. 프로젝트 개요

흑색종(Melanoma)은 피부암 중 가장 치명적인 암으로 조기 진단이 매우 중요합니다. 본 프로젝트는 흑색종과 비흑색종을 딥러닝 기반 CNN 모델을 통해 분류하고, OpenCV를 이용해 병변의 시각적 특징을 정량 분석함으로써 사용자에게 직관적이고 신뢰도 있는 정보를 제공하는 AI 기반 진단 보조 시스템을 개발하는 것을 목표로 했습니다.

---

## 2. 문제 정의 및 목적

- 흑색종은 전체 피부암 중 발생률은 낮지만 사망률이 높아 조기 진단이 필수입니다.
- CNN 기반의 이미지 분석 기술이 피부과 전문의 수준의 정확도를 달성한 최근 연구에 착안
- 분류와 함께 병변의 형태, 색상, 경계 등 시각적 특징을 정량 분석하여 시각화 정보 제공

---

## 3. 데이터셋 및 전처리

### 3.1 데이터 정보
- **출처**: [Kaggle - Melanoma Skin Cancer Dataset](https://www.kaggle.com/datasets/nodariy/melanoma-skin-cancer-dataset-of-10000-images)
- **사용 수량**: 흑색종 1,500장 / 비흑색종 4,500장 (총 6,000장)
- **분할**: Train 70%, Validation 15%, Test 15%
- 
### 3.2 데이터 예시 (흑색종 vs 비흑색종)

| 흑색종 이미지 예시 | 비흑색종 이미지 예시 |
|-------------------|---------------------|
| <img src="assets/sample_melanoma.jpg" width="200"/> | <img src="assets/sample_benign.jpg" width="200"/> |

*Fig. 1. Kaggle Dataset에서 발췌한 실제 피부 병변 이미지 샘플*

> 흑색종은 경계가 불규칙하고 색상이 어두우며, 병변 크기의 증가 속도가 빠른 특성이 있습니다. 이러한 시각적 패턴을 모델이 학습하도록 구성하였습니다.
사용 방법 안내

### 3.3 전처리 코드 예시
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7603, 0.5931, 0.5682],
                         std=[0.1943, 0.1956, 0.2142])
])



## 4. Data Augmentation 전략

### 4.1 적용 배경 및 필요성
- EDA 결과, 병변의 형태, 색상, 크기, 경계 등이 매우 다양함
- 모델의 일반화 성능 향상을 위해 데이터 증강(Augmentation)이 필요함
- 흑색종:비흑색종 비율이 1:3 → 클래스 불균형 해결을 위한 **Weighted Sampling** 적용

---

### 4.2 사용한 Augmentation 기법

| 기법 | 설명 |
|------|------|
| `RandomHorizontalFlip` | 좌우 반전을 통해 병변의 위치 다양성 확보 |
| `RandomRotation` | 병변 방향은 진단에 영향이 적어 회전으로 일반화 가능 |
| `ColorJitter` | 밝기·대비 등 조명을 변화시켜 다양한 환경 학습 가능 |
| `RandomAffine` | 위치, 스케일 변화 적용으로 병변 위치의 일반화 유도 |
| `RandomEqualize` | 피부톤 정규화 및 대비 보정 효과 제공 |

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7603, 0.5931, 0.5682],
                         std=[0.1943, 0.1956, 0.2142])
])

<!-- ============== 4.3 시각화 자료 ============== -->
### 4.3 시각화 자료

| Figure | 설명 |
|--------|------|
| Fig 4-1 | Augmentation 조합(C1~C5)별 테스트 F1-score/Accuracy 비교 막대그래프 |
| Fig 4-2 | Weighted Sampling 적용 전후 성능 향상 곡선 |

<p align="center">
  <img src="assets/fig4_1_augmentation_bar.png" alt="Augmentation 조합별 성능 비교" width="500"/>
  <br><em>Fig 4-1. Augmentation 조합별 성능 비교</em>
</p>

<p align="center">
  <img src="assets/fig4_2_weighted_sampling_curve.png" alt="Weighted Sampling 성능 곡선" width="500"/>
  <br><em>Fig 4-2. Weighted Sampling 적용 전·후 성능 변화</em>
</p>

---

## 5. 모델링

### 5.1 비교 모델 실험

프로젝트에서는 다양한 CNN 기반 모델을 비교 실험하고 최종 모델을 선정하였습니다.

| 모델명         | 장점                                 | 단점                           |
|----------------|--------------------------------------|--------------------------------|
| VGG19          | 구조가 단순하고 구현 용이             | 파라미터 수가 많고 과적합 가능성 있음 |
| ResNet50       | Residual 연결로 학습 안정성 우수      | 표현력 한계 존재               |
| DenseNet121    | 계층 간 feature 재활용 → 일반화 우수  | 학습/추론 속도 다소 느림        |
| EfficientNetB0 | 경량 구조, 높은 효율성                | 구조 최적화 복잡                |
| InceptionV4    | 다양한 크기의 필터 조합으로 특징 추출 | 구조 복잡도 높음               |
| ConvNeXt     | CNN 기반 구조에 Transformer 장점 결합 | 상대적으로 최신 → 리소스 요구 큼  |

---

### 5.2 최종 선정 모델: ConvNeXt

- Meta AI(2022)에서 발표한 최신 CNN 모델로, Vision Transformer에서 착안한 구조
- 핵심 구조:
  - **LayerNorm**, **GELU 활성화 함수**, **Inverted Bottleneck** 구조 사용
  - **Depthwise Convolution** + **1×1 Conv** 조합으로 계산 효율성 확보
- 기존 CNN 대비 더 깊은 표현력과 효율적인 학습 가능

---

### 5.3 학습 전략

| 전략 요소           | 적용 이유 및 효과                                           |
|---------------------|--------------------------------------------------------------|
| Focal Loss          | 클래스 불균형 완화, Hard Negative Sample에 집중             |
| Label Smoothing     | 과적합 및 지나친 확신 방지                                   |
| Warm-up Scheduler   | 초반 학습 속도 조절로 학습 안정성 향상                      |
| CosineAnnealingLR   | Learning Rate 주기적 감소로 수렴 유도                        |
| Test-Time Augmentation (TTA) | 추론 시 다양한 버전의 이미지를 평균하여 성능 향상         |

criterion = FocalLoss()
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

## 6. 성능 평가

### 6.1 주요 평가지표

모델의 성능을 다각도로 평가하기 위해 다음과 같은 지표를 사용하였습니다:

| 지표        | 설명 |
|-------------|------|
| **Accuracy** | 전체 데이터 중 올바르게 예측한 비율 |
| **Precision** | 모델이 양성으로 예측한 것 중 실제 양성의 비율 (False Positive 억제 지표) |
| **Recall** | 실제 양성 중에서 모델이 양성으로 맞춘 비율 (False Negative 억제 지표) |
| **F1-score** | Precision과 Recall의 조화 평균 (불균형 데이터에서 중요) |
| **ROC-AUC** | 모델의 분류 민감도를 평가하는 지표 (0.5~1.0 범위, 클수록 좋음) |

> ⚠ 정확도만으로는 불균형 데이터에서 성능을 오해할 수 있기 때문에, F1-score와 Recall을 특히 중점적으로 평가하였습니다.

---

### 6.2 최종 테스트 결과 (ConvNeXt + C4 + Weighted Sampling)

아래는 가장 우수한 모델의 테스트 데이터셋 성능입니다:

| 지표        | 값     |
|-------------|--------|
| Accuracy    | 96.56% |
| Precision   | 96.63% |
| Recall      | 92.10% |
| F1-score    | 92.84% |
| ROC-AUC     | ≈ 0.97 |

>  이 결과는 `ConvNeXt` 모델에 최적의 `Augmentation 조합(C4)`과 `Weighted Sampling`을 적용한 구성에서 도출되었습니다.

---

### 6.3 시각화 자료 요약

다양한 시각화 자료를 통해 학습 과정을 정량적·정성적으로 평가하였습니다.

-  **Confusion Matrix**  
  → 양성/음성 예측의 정밀도 및 오차 분포 확인

-  **ROC Curve**  
  → 다양한 임계값에 따른 민감도(Sensitivity)와 특이도(Specificity)의 균형 확인

-  **Train vs Validation Loss/Accuracy**  
  → 학습 수렴 정도, 과적합 여부, Early Stopping 타이밍 확인

-  **Precision-Recall Curve**  
  → 희귀 클래스(흑색종)의 정밀도-재현율 균형 확인

> 모든 시각화는 `matplotlib` 및 `seaborn` 기반으로 시각화하였으며, 결과는 모델의 일반화 성능이 우수함을 나타냅니다.

---

### 6.4 실험 결론

- 단순 Accuracy 기준이 아닌 **Recall 및 F1-score 중심의 평가**로 신뢰성 확보
- TTA(Test-Time Augmentation) 적용으로 추론 시 예측 안정성 증가
- EarlyStopping 기준으로 Epoch 4~5에서 가장 일반화 성능 우수

> 전체적으로 학습 안정성, 일반화 성능, 과적합 억제 측면에서 우수한 결과를 확보하였습니다.

<!-- ============== 6.5 결과 시각화 ============== -->
### 6.5 결과 시각화

| Figure | 설명 |
|--------|------|
| Fig 6-1 | Train vs Validation Loss & Accuracy 곡선 |
| Fig 6-2 | Confusion Matrix (Test 셋) |
| Fig 6-3 | ROC-AUC Curve |
| Fig 6-4 | Precision-Recall Curve |

<p align="center">
  <img src="assets/fig6_1_train_val_curve.png" alt="Train/Val Loss & Accuracy" width="550"/>
  <br><em>Fig 6-1. Train-Validation Loss & Accuracy 곡선 (과적합 여부 확인)</em>
</p>

<p align="center">
  <img src="assets/fig6_2_confusion_matrix.png" alt="Confusion Matrix" width="350"/>
  <br><em>Fig 6-2. Confusion Matrix – 클래스별 예측 분포</em>
</p>

<p align="center">
  <img src="assets/fig6_3_roc_auc.png" alt="ROC-AUC Curve" width="450"/>
  <br><em>Fig 6-3. ROC-AUC Curve – 민감도(Sensitivity) vs. 특이도(Specificity)</em>
</p>

<p align="center">
  <img src="assets/fig6_4_precision_recall.png" alt="Precision-Recall Curve" width="450"/>
  <br><em>Fig 6-4. Precision-Recall Curve – 희귀 클래스 평가</em>
</p>

<!-- (선택) Explainable AI 예시 -->
#### 6.5.1 Explainable AI (Grad-CAM)

| Figure | 설명 |
|--------|------|
| Fig 6-5 | 진단 근거 시각화 – Grad-CAM 결과 |

<p align="center">
  <img src="assets/fig6_5_grad_cam.png" alt="Grad-CAM 예시" width="450"/>
  <br><em>Fig 6-5. Grad-CAM으로 확인한 병변 활성 영역</em>
</p>

## 7. 웹 시스템 구현: MelaScan

### 7.1 프로젝트 목적

- 흑색종 진단 모델의 실용적 확장을 위해 Flask 기반 웹 시스템을 구현하였습니다.
- 사용자는 웹 인터페이스를 통해 이미지를 업로드하고, AI 기반으로 분석된 진단 결과 및 병변 특징 정보를 직관적으로 확인할 수 있습니다.

---

### 7.2 핵심 기능

- 이미지 업로드 기능을 통한 사용자 진단 흐름 제공
- 학습된 모델을 기반으로 실시간 흑색종/비흑색종 분류 수행
- 병변의 시각적 특징 분석: ABCD Rule (Asymmetry, Border, Color, Diameter)에 따른 정량 분석 제공
- 분석 결과 및 위험도 판단 결과 시각화
- 진단 결과 저장 기능 구현

---

### 7.3 기술 스택

| 구성 요소 | 사용 기술 |
|------------|------------|
| 프론트엔드 | HTML, CSS, JavaScript, Jinja2 |
| 백엔드     | Flask, PyTorch, timm 라이브러리 |
| 이미지 처리 | OpenCV, Pillow (PIL) |
| 클라이언트 동작 | Fetch API, sessionStorage |

> 프론트엔드와 백엔드는 Flask 템플릿(Jinja2)을 활용하여 자연스럽게 연결되며, 이미지 전처리 및 결과 반환은 PyTorch 기반 모델과 OpenCV가 담당합니다.

---

### 7.4 시스템 페이지 구성

| 경로              | 설명 |
|-------------------|------|
| `/intro`          | 흑색종이 무엇인지와 프로젝트 목적 소개 |
| `/main`           | 사용자가 이미지를 업로드할 수 있는 진입 화면 |
| `/melanoma_result`| 흑색종 진단 결과를 시각화 및 위험도 안내 포함 |
| `/benign_result`  | 비흑색종 진단 결과를 안내하며 추가 위험도 설명은 생략됨 |

---

### 7.5 시스템 아키텍처 흐름

1. 사용자가 웹 인터페이스에서 이미지를 업로드
2. Flask 백엔드가 이미지를 수신하고, 사전 학습된 모델을 통해 분류 실행
3. 분류 결과에 따라 병변의 시각적 특징(ABCD Rule) 분석 및 결과 시각화
4. 분석 결과를 HTML 템플릿에 반영하여 사용자에게 반환

---

### 7.6 향후 확장 계획

- Android Studio 기반의 WebView 앱으로 포팅하여 모바일에서도 사용 가능하도록 구현 예정
- 사용자 진단 이력 저장 및 병원 시스템 연동을 통해 환자-의료진 간 진단 공유 기능 개발 목표

- ## 8. 프로젝트 성과 및 향후 계획

### 8.1 주요 성과

- CNN 기반의 흑색종 분류 모델을 개발하고, 정량적인 병변 분석 기능을 포함한 진단 보조 시스템을 구현하였습니다.
- Augmentation 조합 실험 및 Weighted Sampling 적용을 통해 데이터 불균형 문제를 효과적으로 완화하였습니다.
- ConvNeXt 기반의 모델을 활용하여 F1-score 기준 92.84%, Accuracy 기준 96.56%의 우수한 성능을 달성하였습니다.
- ABCD Rule 기반 병변 분석 로직을 OpenCV로 구현하여, 단순 분류를 넘어 진단에 활용 가능한 병변의 정량 정보까지 제공하는 시스템을 완성하였습니다.
- Flask 기반 웹 시스템을 통해 사용자 이미지 업로드, 진단 수행, 결과 시각화 및 저장 기능까지 통합한 서비스 구조를 설계하고 구현하였습니다.

---

### 8.2 팀워크 및 협업 성과

- 전처리, 모델링, 시스템 구현, 시각화 등 역할을 명확히 분담하여 개발 효율성을 극대화하였습니다.
- 실시간 진단 결과에 대한 해석과 시각화를 통해 사용자 경험 중심의 설계를 도입하였습니다.
- 학습 중간 결과를 바탕으로 전략을 수시로 점검하고 모델 구조, 하이퍼파라미터, 데이터 증강 전략을 공동으로 개선하였습니다.

---

### 8.3 향후 개선 및 확장 방향

- 사용자 이력 저장 기능 도입을 통해 반복 진단 및 사용자 맞춤 피드백 기능 개발 예정
- Android Studio 기반의 WebView를 활용하여 모바일 앱으로 확장하고, 오프라인 환경에서도 이미지 진단이 가능하도록 구현
- 병원 서버와의 연동 기능을 통해 의료진과의 커뮤니케이션을 원활하게 하고, 실제 임상 환경에서 활용 가능한 진단 보조 시스템으로 확장 예정
- Explainable AI 기반의 결과 해석 기능 추가 예정 (예: Grad-CAM 시각화)으로 모델의 판단 근거를 사용자에게 설명하는 기능 강화

## 9. 참고문헌

1. Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017).  
   Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542(7639), 115–118.  
   https://doi.org/10.1038/nature21056

2. 김용수, 조우성, 오승민, 조효은, 백용선. (2023).  
   합성곱 신경 회로망 모델을 활용한 흑색종 피부암 진단. *한국지능시스템학회 논문지*, 33(3), 228-241.  
   https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11444880

3. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021).  
   An image is worth 16x16 words: Transformers for image recognition at scale. *International Conference on Learning Representations (ICLR)*.  
   https://arxiv.org/abs/2010.11929

4. Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022).  
   A ConvNet for the 2020s. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 11976–11986.  
   https://arxiv.org/abs/2201.03545

5. Nodariy. (2022).  
   Melanoma Skin Cancer Dataset of 10,000 Images [Data set]. *Kaggle*.  
   https://www.kaggle.com/datasets/nodariy/melanoma-skin-cancer-dataset-of-10000-images
