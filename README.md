# 🎯 VGG19 Transfer Learning & Model Compression Project

VGG19 모델의 **전이학습(Transfer Learning)** 효과를 검증하고, **모델 경량화(Model Compression)** 를 통해 적은 데이터로 효율적인 이미지 분류 모델을 구축한 프로젝트입니다.

---

## 📌 프로젝트 개요

### 목표
- VGG19 논문 기반 모델 구현 및 전이학습 효과 비교
- 적은 데이터셋(~900장)에서 높은 성능 달성
- 모델 경량화를 통한 효율성 개선

### 데이터셋
- **클래스**: Santa / Normal (2-class classification)
- **데이터 구성**: Train (~900장) / Validation / Test
- **전처리**: 224×224 리사이즈, 정규화, 데이터 증강

---

## 🔬 실험 1: VGG19 전이학습 효과 검증

### 1️⃣ 논문 기반 VGG19 구현 (From Scratch)

**모델 구조**:
```python
class VGG19(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG19, self).__init__()
        self.features = nn.Sequential(
            # 5개 Conv Block (16개 Conv Layer)
            # 64 → 128 → 256 → 512 → 512
            # 각 블록마다 MaxPooling
            ...
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
```

**결과**:
- ❌ **학습 실패**: Loss = NaN, Accuracy ≈ 50% (랜덤 수준)
- **원인**: 데이터 부족 (~900장) + 파라미터 과다 (1억+)

---

### 2️⃣ 전이학습 적용 (Transfer Learning)

**전략**:
```python
# ImageNet 사전 학습 가중치 로드
weights = VGG19_Weights.IMAGENET1K_V1
model = vgg19(weights=weights)

# Features (Conv layers) 고정
for param in model.features.parameters():
    param.requires_grad = False

# Classifier만 재정의 (2-class)
model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(4096, 2)
)
```

**결과**:
- ✅ **Epoch 1**: 91.76% → **최종**: ~97% 정확도
- Optimizer 비교:
  - **Adam**: 안정적 학습, 97% 달성
  - **SGD**: Epoch 12 이후 발산 (Gradient Exploding)

---

## 🚀 실험 2: 모델 경량화 (Model Compression)

### VGG19 경량화 버전 구현

**설계 전략**:
- 채널 수 감소: 64→128→256→512 → **32→64→128→256**
- Depth 감소: 5개 블록 → **4개 블록**
- BatchNorm 추가로 학습 안정성 확보

**BatchNormalization 적용 이유**:
- 각 층의 입력 분포를 정규화하여 학습 안정성 향상
- 더 높은 Learning Rate 사용 가능
- Gradient Vanishing/Exploding 방지 효과
- 경량화 모델의 성능 저하 보완

**모델 구조**:
```python
class VGG19_Small(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 32 channels
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2: 64 channels
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3: 128 channels
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4: 256 channels
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
```

**경량화 효과**:
- 파라미터 수: ~95% 감소
- 학습 속도: 대폭 개선
- 성능: 유사하거나 향상된 정확도 유지

---

## 📊 실험 결과 비교

| 모델 | Accuracy | 학습 안정성 | 파라미터 수 |
|------|----------|-------------|-------------|
| VGG19 (From Scratch) | ~50% (실패) | ❌ NaN | ~138M |
| VGG19 (Transfer Learning) | ~97% | ✅ 안정 | ~138M |
| VGG19_Small (경량화) | ~95-97% | ✅ 안정 | ~7M |

**핵심 인사이트**:
1. **전이학습의 중요성**: 적은 데이터에서 필수적
2. **Optimizer 선택**: Adam이 SGD보다 안정적
3. **경량화 효과**: 성능 유지하며 효율성 95% 개선

---

## 🛠️ 기술 스택

- **Framework**: PyTorch
- **Libraries**: torchvision, OpenCV, matplotlib, tqdm
- **Environment**: Google Colab (GPU)

---


## 🎓 주요 학습 내용

### 1. 전이학습 (Transfer Learning)
- ImageNet 사전 학습 가중치 활용
- Feature Extraction vs Fine-tuning 전략
- 적은 데이터셋에서의 효과적인 학습

### 2. 모델 최적화
- Optimizer 비교 (SGD vs Adam)
- Learning rate, momentum 하이퍼파라미터 조정
- Gradient Exploding 문제 해결

### 3. 모델 경량화
- 채널 수/Depth 조정을 통한 파라미터 감소
- BatchNormalization을 통한 학습 안정성 확보
- 성능과 효율성의 트레이드오프 분석

---

## 📈 향후 개선 방향

- [ ] Learning Rate Scheduler 적용
- [ ] Early Stopping 구현
- [ ] 다른 경량화 기법 실험 (Knowledge Distillation, Pruning)

---


## 👤 Author

**팀-라이언일병구하기**
- 딥러닝 모델 구현 및 최적화
- 전이학습 효과 검증 실험
