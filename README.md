# onlyextrinsic_vanilla — **Extrinsic Turbo** LDPC + EDM Source Denoiser

> **이 저장소는 "extrinsic turbo" 방식이다.**
>
> Fashion-MNIST 이미지를 5G LDPC로 전송하는 체인에, EDM 기반 소스 디노이저를
> **터보(extrinsic) 형태**로 결합한 실험이다. BP(코드 제약)와 denoiser(이미지 prior)를
> 각각의 **extrinsic**으로 교환하며 반복 디코딩한다.
>
> 이 방식은 **Expectation Propagation(EP)식 cavity/site 접근을 채택하지 않는다.**
> site를 저장하거나 cavity 나눗셈을 수행하지 않으며, 대신 매 청크에서 고정된
> `channel` LLR을 base로 놓고 extrinsic을 가산한다. 두 방식의 정확한 대비는
> [§4 Extrinsic Turbo vs 대안적 EP식 접근](#4-extrinsic-turbo-vs-대안적-ep식-접근-가장-중요)을 참조.
>
> 성능 대표값은 **200-round 실측**만 인용한다 (§6). σ는 이 브랜치에서 **0.3 고정**이며
> adaptive sigma가 없다.

핵심 목표는 **동일 총 BP iteration의 baseline BP 대비 BLER/BER 개선**이다.

---

## 1) 시스템 개요

두 개의 프레임워크를 결합한다.

- **TensorFlow + Sionna**: 채널 코딩 / 변조 / AWGN / LDPC BP 디코딩
- **PyTorch**: 이미지 prior 기반 EDM(score) 디노이저

입출력 흐름:

1. Fashion-MNIST test 이미지(28×28)를 비트열(6272 bits = 784 px × 8 bpp)로 변환
2. CRC24A 추가 후 5G LDPC 인코딩 (`N=12600`)
3. BPSK(PAM-1) 변조 후 AWGN 채널 통과
4. APP demapper로 채널 LLR 생성
5. `LDPC5GDecoder_soft`가 **BP 청크 반복 + denoiser extrinsic 결합** (본 저장소의 핵심)
6. CRC 기반 BLER, 비트 비교 기반 BER 계산

---

## 2) 핵심 업데이트 식 (Extrinsic Turbo)

BP 스케줄은 콤마로 나뉜 청크 리스트다 (예: `[5,5,5,5,5,5]` = 5 iter 짜리 BP 청크 6개).
각 청크의 BP 이후(**마지막 청크 제외**) 다음을 계산한다.

```text
bp_ext    = BP_post  − payload_intr        # BP extrinsic  (turbo BP)
src_ext   = src_post − BP_post             # source extrinsic (turbo source)

new_input = channel + β·bp_ext + α_t·src_ext
```

- `channel` (`payload0`): 초기 채널 intrinsic LLR — **고정(frozen) base**
- `payload_intr`: **해당 청크 BP에 실제로 입력된** a priori LLR
- `BP_post`: 해당 청크 BP posterior
- `src_post`: denoiser가 `BP_post`를 입력받아 낸 posterior logits
- `α_t`: 청크별 alpha schedule 값 (§5)
- `β`: BP extrinsic 가중치 (0.1 근처)

### 구조: 매 청크 channel에서 다시 시작해 extrinsic을 쌓는다

이 방식의 핵심은 `new_input`이 **매번 고정 `channel`을 base로 재구성**된다는 점이다.
site를 누적 저장하지 않는다. 각 청크는 `channel`에서 출발해 그 위에 두 종류의
extrinsic을 얹는다:

- **`bp_ext = BP_post − payload_intr`** — BP가 **자기 입력(`payload_intr`) 대비** 새로
  알아낸 것. 이것이 turbo BP extrinsic이다. `payload_intr`은 바로 이 청크 BP에 넣은
  값이므로, 그것을 빼면 이 청크에서 BP가 새로 기여한 정보만 남는다.
  - 첫 청크에서는 `payload_intr == channel`이므로 `bp_ext = BP_post − channel`.
  - 이후 청크에서는 `payload_intr`이 이미 이전 extrinsic들을 담고 있으므로,
    그것을 빼면 **더 깨끗한(cleaner) extrinsic**이 된다.
- **`src_ext = src_post − BP_post`** — denoiser가 **자기 입력(`BP_post`) 대비** 새로
  알아낸 것. 이것이 turbo source extrinsic이다. denoiser의 a priori 입력이
  정확히 `BP_post`이므로 교과서적 `posterior − a_priori` 형태와 일치한다.
  denoiser는 비선형이라 `src_post − BP_post`는 참 source extrinsic의 **표준적
  실용 근사**다.

즉 BP extrinsic과 source extrinsic을 **분리해 동시에** `channel` 위로 반영한다.

### 적용 범위: extrinsic 되먹임은 payload(이미지) 비트에만

`payload_intr`은 **payload(이미지) 비트 `k_payload = 6272`개에만** 해당한다. BP 입력은
`llr_bp = concat([payload_intr, crc_and_rest, filler, parity])` 형태로 재구성되는데,
extrinsic 갱신을 받는 것은 앞의 payload 슬라이스뿐이고 **CRC24 비트(24개)·filler·parity는
매 청크 channel LLR 값으로 고정**된다. `bp_ext`/`src_ext`도 payload 슬라이스 위에서만
계산된다. (denoiser는 이미지 prior이므로 이미지 비트에만 작용하는 것이 자연스럽다.)

### 특수 케이스 (코드 주석과 일치)

| α | β | `new_input` | 의미 |
|---|---|---|---|
| 0 | 0 | `channel` | baseline BP (denoiser 미사용) |
| 0.1 | 0 | `channel + 0.1·src_ext` | denoiser-only 보정 |
| 0 | 1 | `BP_post` | turbo BP 되먹임 |

### 구현 대조 (numerically verified)

이 갱신식은 실제 `decoder.py` 구현과 **수치적으로 정확히 일치**함을 확인했다. 소규모
배치에서 BP 입출력과 denoiser 입출력을 계측해 대조한 결과:

- chunk-0의 BP a priori 입력 **== channel**(frozen base),
- 매 denoiser 호출의 입력 **== `BP_post`**,
- `payload_intr[i+1]` **== `channel + β·bp_ext[i] + α_t·src_ext[i]`** (모든 청크에서 `max|Δ|=0`),
- base가 청크 진행 중 `channel`로 고정(site 누적 없음),
- alpha 인덱싱: `i`번째 denoiser 호출이 `alpha_schedule[i]` 사용.

즉 `decoder.py`의 루프([`call()`](decoder.py))는 위 식 그대로다.

---

## 3) 왜 turbo extrinsic인가 (설계 의도)

denoiser는 **불완전**하다 (test set MSE ≈ 0.1) 그리고 확신이 과한 경향이 있다.
source를 과하게 믿으면 BP 수렴이 오염된다. 그래서 이 저장소는:

- BP와 source의 기여를 **각각 extrinsic으로 분리**해 가중치(β, α_t)로 독립 제어하고,
- **α schedule로 청크가 진행될수록 source 가중치를 감쇄**해(§5), 최종적으로 BP(코드
  제약)가 안착하도록 유도한다.

이는 의도적인 선택이며, EP식 site/cavity 메커니즘 없이 turbo extrinsic만으로 구성한다.

---

## 4) Extrinsic Turbo vs 대안적 EP식 접근 (가장 중요)

이 저장소는 **turbo extrinsic** 방식을 쓴다. 개념적으로 존재하는 대안은
**Expectation Propagation(EP)식 cavity/site** 방식인데, **이 저장소는 그것을 채택하지
않는다.** 아래에서 두 방식을 정확히 대비한다. (혼동 방지: 이 저장소의 방식을 "EP"라고
부르지 말 것.)

### (a) 업데이트 구조

| | **이 저장소 (Extrinsic Turbo)** | **대안: EP식 (cavity/site)** |
|---|---|---|
| 갱신식 | `new_input = channel + β·bp_ext + α·src_ext` | `posterior = cavity + new_site` |
| base | 매 청크 **고정 `channel`** | 직전 **posterior** |
| 저장 상태 | 없음 (extrinsic만 그때그때 계산) | **각 factor의 site를 명시적으로 저장** |
| 갱신 절차 | channel 위에 extrinsic을 **가산** | cavity 생성 후 **site를 교체** |

이 저장소는 site를 저장하지 않고, 매 청크 `channel`에서 다시 시작해 extrinsic을
쌓는다. EP식 대안은 각 factor의 site를 유지하며 posterior에서 그 site를 떼어
cavity를 만든 뒤, 갱신된 site로 교체한다.

### (b) extrinsic vs cavity — 핵심 구분

이 둘은 "자기 기여를 어떻게 제거하는가"에서 갈린다.

- **이 저장소의 `src_ext = src_post − BP_post` = turbo extrinsic.**
  denoiser의 입력(`BP_post`)을 빼서 "이번 기여"를 추출한다. 이는 **자기 되먹임의
  부분적 제거(partial self-removal)**다. `BP_post` 안에는 이전 청크에서 온 source
  기여가 섞여 있을 수 있으므로, 이전 source 기여가 **일부 남을 수 있다.**

- **EP식 대안의 cavity = posterior − (그 factor의 직전 저장된 site).**
  자기 기여를 **정확히(exactly) 완전 제거**한다. 이를 위해 **site 저장이 필수**다.

정리하면:

- turbo extrinsic은 **"이번 입력(`BP_post`)을 빼는"** 방식 → 부분적 self-removal,
  이전 source 기여가 일부 잔존할 수 있음.
- EP cavity는 **"저장된 자기 site를 빼는"** 방식 → 정확한 제거, 대신 site 저장 필요.

**이 저장소는 의도적으로 turbo extrinsic을 쓴다.** site 저장이나 cavity 나눗셈
메커니즘이 없다.

### (c) 이 저장소에 **없는** 것 (turbo이므로)

- ❌ 명시적 site 저장 없음
- ❌ cavity 나눗셈 없음 — `BP_post` 빼기로 self-removal을 **근사**
- ✅ 대신 **α schedule**로 청크별 source 가중치를 조절 (§5)

---

## 5) Alpha Schedule (이 저장소의 핵심 기법)

```text
alpha_schedule = [0.10, 0.08, 0.06, 0.04, 0.02]
```

청크마다 다른 α로 source extrinsic을 반영한다: **초반 강하게(0.10) → 후반 약하게(0.02).**

**왜 후반 감쇄인가.**
denoiser는 불완전(test set MSE ≈ 0.1)하고 확신이 과하므로, source를 과하게 믿으면
BP 수렴이 오염된다. 후반으로 갈수록 source 가중치를 낮춰 **BP(code 제약)가 최종적으로
안착**하도록 유도한다. (초반엔 denoiser의 이미지 prior로 크게 끌어당기고, 마무리는
코드 제약이 지배하게 두는 구조.)

**β (BP extrinsic 가중치)** 는 0.1 근처를 쓴다.

### `decoder.py`의 alpha_schedule 길이 처리 — 스케줄 변경 시 주의

denoiser 호출 수 `n_denoiser = len(bp_schedule) − 1`이다 (마지막 청크는 denoiser를
호출하지 않으므로). alpha_schedule 길이가 이와 다르면:

| 상황 | 동작 | 로그 |
|---|---|---|
| schedule이 **짧으면** (`< n_denoiser`) | **마지막 값을 반복** | `[INFO]` |
| schedule이 **길면** (`> n_denoiser`) | **초과분을 truncate** | `[WARN]` |

예: `bp_schedule=[5,5,5,5,5,5]` → 청크 6개 → denoiser 호출 5회 → alpha_schedule은
5개 필요. `[0.10,0.08,0.06,0.04,0.02]`가 정확히 맞는다. **스케줄을 바꿀 때 이 반복/절단
동작에 주의**할 것.

---

## 6) 현재 Best 검증 결과 (200-round 실측, 인용용)

아래는 **최근 실행한 `stats_bler_ebno08.py`의 200-round 실측 결과**다
(`results/bler_stats_ebno08.json`).

**조건:**
- Eb/N0 = **0.8 dB**
- Proposed schedule: `[5,5,5,5,5,5]`
- alpha schedule: `[0.10,0.08,0.06,0.04,0.02]`
- β = 0.1, **σ = 0.3 (고정, adaptive sigma 없음)**
- batch = 64, rounds = **200** → **총 12,800 blocks/decoder**

| Decoder | ACK / NACK | Pooled BLER | 95% CI (Wilson) | Pooled BER |
|---|---:|---:|---:|---:|
| BP-30 (Baseline) | 5996 / 6804 | 0.531563 | [0.522910, 0.540197] | 3.8853e-03 |
| **Proposed [5×6]** | 12780 / **20** | **0.001563** | **[0.001012, 0.002412]** | **7.7228e-07** |

**요약:**
- Eb/N0=0.8 dB에서 Proposed는 baseline 대비 BLER이 약 **340배** 낮다
  (0.531563 / 0.001563 ≈ 340).
- 본 저장소 기준 **현재 best 검증 수치**는 `Proposed BLER = 0.001563` (NACK 20/12800).

**이전 20-round 수치에 대한 정정.**
과거 README에는 20-round 수치(`BLER 0.003125`, NACK 4개)가 실려 있었다. 이 수치는
**샘플이 적어(총 1280 blocks) CI가 넓고 통계적으로 불안정**하여 대표값으로 부적절했다.
따라서 위 **200-round 수치로 대체**한다. 200-round는 **NACK 20개로 CI가 좁아
(±0.0007 수준) 통계적으로 유의미**하다.

---

## 7) 측정 방법론 (이 저장소의 강점)

- **Wilson score 95% 신뢰구간**으로 BLER의 이항비율 CI를 산출한다 (극소 오류율에서도
  타당한 구간). `stats_bler_ebno08.py`의 `wilson_ci()`.
- **Pooled BLER**(전체 블록을 모아 계산)과 **per-round BLER**(라운드별 평균±표준편차)를
  **분리**해 보고한다.
- **공정 비교**: baseline은 **동일 총 BP iteration** 기준(Proposed의 `[5×6]` = 30 iter
  ↔ **BP-30**)으로 맞춘다.
- **best 검증 스크립트는 `stats_bler_ebno08.py`** 다.

---

## 8) 저장소 구조 (핵심 파일)

- `decoder.py` — `LDPC5GDecoder_soft`: **BP + denoiser extrinsic turbo 루프** (§2 갱신식,
  alpha_schedule 길이 처리 §5)
- `denoiser.py` — TF↔PyTorch 브리지 `SoftDenoiser`. 항상 **순수 source extrinsic
  (`src_post − BP_post`)** 만 반환 (BP/source 가중치는 decoder의 α/β가 담당)
- `source_prior.py` — LLR↔이미지 변환 + EDM posterior + `compute_source_extrinsic`
- `score_denoiser/networks.py` — EDM `SongUNet`/`EDMPrecond` 백본
- `train_denoiser.py` — denoiser 학습, `checkpoints/denoiser.pt` 생성
- **`stats_bler_ebno08.py`** — Eb/N0=0.8 BLER 통계 + Wilson CI (**best 검증 스크립트**)
- `plot_comparison.py`, `plot_comparison_poster.py` — baseline vs proposed Eb/N0 곡선
- `vanilla_sweep.py` — β / alpha schedule / EbN0 스윕, CSV 누적
- `sigma_sweep.py` — sigma 스윕 / `safe_sweep.sh` — 장시간 스윕 배치
- `analyze_warmup.py` — warm-up 스윕 결과 분석
- `visualize_progression.py`, `visualize_progression_clean.py` — 디코딩 단계 시각화

---

## 9) 빠른 실행 가이드

작업 디렉토리:

```bash
cd /home/LJH/onlyextrinsic_vanilla
```

### 9.1 디노이저 학습 (선택)

```bash
python train_denoiser.py --epochs 5 --batch 128 --lr 2e-4
```
출력: `checkpoints/denoiser.pt`

### 9.2 메인 비교 플롯

```bash
CUDA_VISIBLE_DEVICES=0 python plot_comparison_poster.py \
  --bp-schedule 5 5 5 5 5 5 \
  --alpha-schedule "0.10,0.08,0.06,0.04,0.02" \
  --beta 0.1 --sigma 0.3 \
  --ebno 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 \
  --baselines 30 --batch 64 --rounds 5 \
  --out results/comparison_5x6_beta01.png
```
출력: `results/comparison_5x6_beta01_bler.png`, `..._ber.png`

### 9.3 Eb/N0=0.8 BLER 통계 (현재 best 검증, 200-round)

```bash
CUDA_VISIBLE_DEVICES=0 python stats_bler_ebno08.py \
  --ebno 0.8 --batch 64 --rounds 200
```
출력: `results/bler_stats_ebno08.json`, `results/bler_stats_ebno08.csv`

---

## 10) 결과 파일 형식

### `results/vanilla_sweep.csv` 컬럼
```text
timestamp,tag,mode,ebno_db,sigma,beta,alpha_schedule,
bler,ber,n_blocks,n_acks,n_nacks,n_bit_errors,bp_schedule
```

### `results/bler_stats_ebno08.json`
- `config`: 실행 파라미터 (ebno, bp_schedule, alpha_schedule, beta, sigma, batch, rounds)
- `results`: 디코더별 BLER/BER/Wilson CI/라운드 통계

---

## 11) 의존성

- Python 3.8+
- TensorFlow 2.x, Sionna
- PyTorch, torchvision
- numpy, matplotlib

---

## 12) 주의사항

- **이 방식은 extrinsic turbo다** — "EP"라고 부르지 말 것. site 저장/cavity 나눗셈 없음 (§4).
- **σ는 이 브랜치에서 0.3 고정** — adaptive sigma 메커니즘 없음.
- **성능 대표값은 200-round 실측만 인용** (§6). 20-round 수치는 통계 불안정으로 폐기.
- `checkpoints/denoiser.pt`가 없으면 denoiser가 랜덤 가중치로 동작한다 (`[WARN]` 출력).
- BLER/BER는 `batch × rounds` 샘플 수에 의존하므로, 논문급 통계엔 `rounds`를 충분히 키운다
  (본 best는 rounds=200, NACK 20개).
- 현재 파이프라인은 TF↔PyTorch 브리지를 거치므로 성능 최적화 여지가 있다.
