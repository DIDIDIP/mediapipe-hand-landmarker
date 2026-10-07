# 📘 강의 노트: MediaPipe Hand Landmarker로 웹캠 손 인식하기

> **작성일:** 2026-10-07
> **실습 환경:** Windows 11 Pro / Python 3.14.8 / mediapipe 1.1.0 / OpenCV 5.0.0
> **참고 자료:** [Google AI Edge – 손 특징점 감지 가이드](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)

---

## 🎯 학습 목표

이 강의를 마치면 다음을 할 수 있습니다.

1. MediaPipe **Hand Landmarker**가 무엇이고 어떤 결과를 주는지 설명할 수 있다.
2. 공식 가이드 페이지에서 **모델 번들(`.task`)** 을 찾아 내려받을 수 있다.
3. Python으로 **웹캠 실시간 손 관절 인식** 프로그램을 작성하고 실행할 수 있다.

---

## 1교시 — 개념 잡기: Hand Landmarker란?

MediaPipe Hand Landmarker는 **이미지 속 손을 찾아 관절 위치(랜드마크)를 알려주는** 작업(Task)입니다.

### 입력과 출력

| 작업 입력 (셋 중 하나) | 작업 출력 |
|---|---|
| 정지 이미지 | 감지된 손의 **손잡이(handedness)** — 왼손/오른손 |
| 디코딩된 비디오 프레임 | **이미지 좌표계** 기준 손 랜드마크 (0~1로 정규화) |
| 실시간 영상 피드 (웹캠) | **세계 좌표계** 기준 손 랜드마크 (미터 단위 3D) |

### 손 하나 = 21개 점

모델은 손 하나당 **21개 손가락 관절 좌표**를 반환합니다.

```
            8   12  16  20      ← 손가락 끝 (검지·중지·약지·새끼)
            |   |   |   |
            7   11  15  19
            |   |   |   |
     4      6   10  14  18
      \     |   |   |   |
       3    5 - 9 - 13- 17
        \   |          /
         2  |         /
          \ |        /
           1|       /
             0 ----            ← 0번 = 손목(WRIST)
```

| 번호 | 부위 | 번호 | 부위 |
|---|---|---|---|
| 0 | 손목 | 9–12 | 중지 (MCP→끝) |
| 1–4 | 엄지 (CMC→끝) | 13–16 | 약지 |
| 5–8 | 검지 (MCP→끝) | 17–20 | 새끼손가락 |

> 💡 **기억하기:** 손가락 끝은 **4의 배수**(4, 8, 12, 16, 20)입니다.

### 모델은 2단계 구조

`hand_landmarker.task` 하나에 **두 모델이 함께 묶여(bundle)** 있습니다.

1. **손바닥 감지 모델 (Palm Detection)** — 이미지 전체에서 손이 *어디* 있는지 찾음
2. **손 랜드마크 모델 (Hand Landmarks)** — 찾은 영역 안에서 21개 점을 정밀하게 찍음

> 📝 **메모:** 비디오/라이브 모드에서는 매 프레임 손바닥 감지를 다시 하지 않습니다.
> 이전 프레임 손 위치로 **추적(tracking)** 하다가, 신뢰도가 떨어질 때만 손바닥 감지를 다시 돌립니다 → 그래서 빠릅니다.

| 모델명 | 입력 크기 | 양자화 | 비고 |
|---|---|---|---|
| HandLandmarker (full) | 192×192, 224×224 | float16 | 약 3만 장의 실제 이미지 + 합성 데이터로 학습 |

---

## 2교시 — 준비하기: 모델 & 라이브러리

### Step 1. 모델 내려받기

가이드 페이지 → **"모델"** 섹션 → 표의 **"버전: 최신"** 링크를 클릭하면 `hand_landmarker.task`가 다운로드됩니다.
(오늘 실습에서는 Claude in Chrome으로 페이지를 열어 직접 클릭해서 받았습니다.)

- 파일명: `hand_landmarker.task`
- 크기: **7,819,105 bytes (약 7.8 MB)**
- 위치: 실행 스크립트와 **같은 폴더**에 둡니다.

> 터미널로 받고 싶다면 아래 명령도 같은 파일을 받습니다.
> ```
> curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
> ```

### Step 2. 라이브러리 설치

```bash
python -m pip install mediapipe opencv-python
```

설치 확인:

```bash
python -c "import mediapipe as mp, cv2; print(mp.__version__, cv2.__version__)"
# 1.1.0 5.0.0
```

> ⚠️ **실습 중 만난 경고:** `cffi-gen-src.exe ... is not on PATH`
> → 설치는 성공한 것이고, 스크립트 실행 파일 경로 안내일 뿐이라 **무시해도 됩니다.**

---

## 3교시 — 핵심 설정값 (Configuration Options)

| 옵션 | 설명 | 값 범위 | 기본값 |
|---|---|---|---|
| `running_mode` | 실행 모드 | `IMAGE` / `VIDEO` / `LIVE_STREAM` | `IMAGE` |
| `num_hands` | 최대 감지할 손 개수 | 1 이상 정수 | 1 |
| `min_hand_detection_confidence` | 손바닥 감지 성공 최소 점수 | 0.0 ~ 1.0 | 0.5 |
| `min_hand_presence_confidence` | 랜드마크 모델의 "손이 있다" 최소 점수 | 0.0 ~ 1.0 | 0.5 |
| `min_tracking_confidence` | 추적 성공 최소 점수 (이전·현재 프레임 IoU) | 0.0 ~ 1.0 | 0.5 |
| `result_callback` | 비동기 결과 수신 함수 | 함수 | — (`LIVE_STREAM`에서만) |

### 실행 모드 비교 — 어떤 걸 고를까?

| 모드 | 호출 함수 | 특징 | 언제 쓰나 |
|---|---|---|---|
| `IMAGE` | `detect(image)` | 한 장씩 독립 처리 | 사진 파일 |
| `VIDEO` | `detect_for_video(image, ts_ms)` | **동기식**, 프레임 간 추적 사용 | 동영상 파일, 간단한 웹캠 |
| `LIVE_STREAM` | `detect_async(image, ts_ms)` | **비동기식**, 콜백으로 결과 수신 | 지연에 민감한 실시간 앱 |

> ✅ **오늘의 선택: `VIDEO` 모드**
> 웹캠도 결국 "연속된 프레임"이므로 VIDEO 모드로 충분합니다.
> 결과가 함수 반환값으로 바로 오니 콜백·스레드 동기화가 필요 없어 **입문용으로 가장 단순**합니다.
> 단, `timestamp_ms`는 **반드시 단조 증가**해야 합니다 → `time.monotonic()` 사용.

---

## 4교시 — 코드 실습: `hand_webcam.py`

### 전체 흐름

```
웹캠 프레임 읽기 (BGR)
      ↓  좌우 반전 (거울 모드)
BGR → RGB 변환
      ↓
mp.Image로 감싸기
      ↓
landmarker.detect_for_video(image, timestamp_ms)
      ↓
결과(21개 점 × 손 개수) → OpenCV로 그리기
      ↓
화면 출력, q/ESC 누르면 종료
```

### 코드 해설 (핵심 부분)

**① 모델 옵션 만들기**

```python
options = vision.HandLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=args.hands,                 # 기본 2개
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

**② 프레임마다 추론하기**

```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # OpenCV는 BGR, MediaPipe는 RGB!
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
timestamp_ms = int((time.monotonic() - start) * 1000)  # 단조 증가 타임스탬프
result = landmarker.detect_for_video(mp_image, timestamp_ms)
```

**③ 결과 그리기** — 정규화 좌표(0~1)를 픽셀 좌표로 바꾸는 것이 포인트

```python
points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

for conn in HAND_CONNECTIONS:          # 관절 연결선 (뼈대)
    cv2.line(frame, points[conn.start], points[conn.end], (0, 255, 0), 2)
for x, y in points:                    # 관절 점
    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
```

**④ 결과 객체 구조 (`HandLandmarkerResult`)**

```python
result.hand_landmarks        # [손][21개 점] → .x, .y, .z (정규화)
result.hand_world_landmarks  # [손][21개 점] → 미터 단위 3D 좌표
result.handedness            # [손][카테고리] → .category_name ("Left"/"Right"), .score
```

### 화면에 보이는 것

| 요소 | 색 | 의미 |
|---|---|---|
| 초록 선 | (0, 255, 0) | 관절 연결선 |
| 빨간 점 | (0, 0, 255) | 21개 랜드마크 |
| 하늘색 글자 | (255, 255, 0) | `Left 0.97` 등 손 구분 + 신뢰도 |
| 흰 글자 (좌상단) | (255, 255, 255) | FPS, 감지된 손 개수 |

---

## 5교시 — 실행 & 결과

### 실행 방법

```bash
cd C:\Users\USER\Downloads\mediapipe
python hand_webcam.py                       # 기본 카메라(0), 최대 손 2개
python hand_webcam.py --camera 1 --hands 1  # 다른 카메라, 손 1개
```

- 종료: 창을 클릭한 뒤 **`q`** 또는 **`ESC`**

### 오늘 실습 기록

| 단계 | 결과 |
|---|---|
| 모델 다운로드 | ✅ `hand_landmarker.task` 7.8 MB |
| 라이브러리 설치 | ✅ mediapipe 1.1.0, OpenCV 5.0.0 |
| 빈 이미지 추론 테스트 | ✅ 모델 로드 정상, 감지된 손 0개 (정상) |
| 웹캠 실행 | ✅ "MediaPipe Hand Landmarker" 창 정상 표시, 종료 코드 0 |

### 실행 시 나오는 로그 — 걱정하지 마세요

```
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 ... inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature ...
W0000 ... landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS ...
```

> 📝 **메모:** 모두 MediaPipe 내부 정보/경고 메시지입니다. CPU 가속(XNNPACK)이 켜졌다는 안내와 내부 계산 방식 안내일 뿐, **동작에는 영향이 없습니다.**

---

## 🛠️ 트러블슈팅 Q&A

| 증상 | 원인 / 해결 |
|---|---|
| `Cannot open camera 0` | 다른 앱(Zoom, Teams 등)이 카메라 사용 중 → 종료 후 재시도, 또는 `--camera 1` |
| 창이 안 보임 | 작업 표시줄에서 "MediaPipe Hand Landmarker" 창 찾기 |
| 손을 잘 못 잡음 | 조명을 밝게, 손 전체가 화면에 들어오게, `min_*_confidence` 값을 0.3 정도로 낮추기 |
| 왼손/오른손이 반대로 나옴 | `cv2.flip(frame, 1)` 거울 반전 여부에 따라 달라짐 → 반전을 끄면 반대로 표시 |
| `timestamp must be monotonically increasing` 에러 | 같은/작은 타임스탬프를 넣은 것 → `time.monotonic()` 기반으로 계산 |
| 모델 파일 못 찾음 | `hand_landmarker.task`가 `hand_webcam.py`와 같은 폴더에 있는지 확인 |

---

## ✍️ 복습 & 과제

**복습 퀴즈**

1. 손 하나에서 반환되는 랜드마크 개수는? → <details><summary>정답</summary>21개</details>
2. 검지 손가락 끝의 인덱스 번호는? → <details><summary>정답</summary>8번</details>
3. OpenCV 프레임을 MediaPipe에 넣기 전 꼭 해야 하는 변환은? → <details><summary>정답</summary>BGR → RGB</details>
4. `VIDEO` 모드와 `LIVE_STREAM` 모드의 가장 큰 차이는? → <details><summary>정답</summary>동기식(반환값) vs 비동기식(콜백)</details>

**도전 과제**

- [ ] 검지 끝(8번) 좌표를 화면에 숫자로 표시해 보기
- [ ] 엄지 끝(4) ↔ 검지 끝(8) 거리를 계산해서 "집기(pinch)" 제스처 감지하기
- [ ] 펴진 손가락 개수 세기 (손가락 끝 y < 두 번째 관절 y 이면 펴짐)
- [ ] `LIVE_STREAM` 모드 + `result_callback`으로 코드 바꿔 보기

---

## 📁 폴더 구성

```
mediapipe/
├── README.md              ← 이 강의 노트
├── hand_webcam.py         ← 웹캠 손 인식 실행 코드
└── hand_landmarker.task   ← 모델 번들 (손바닥 감지 + 랜드마크, 7.8 MB)
```
