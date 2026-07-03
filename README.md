# onlyextrinsic_vanilla

Fashion-MNIST 이미지를 5G LDPC로 전송하는 체인에, EDM 기반 소스 디노이저를 터보(extrinsic) 형태로 결합한 실험 저장소입니다.  
핵심 목표는 **Baseline BP 대비 BLER/BER 개선**입니다.

---

## 1) 시스템 개요

이 프로젝트는 아래 2개 프레임워크를 결합합니다.

- **TensorFlow + Sionna**: 채널 코딩/변조/AWGN/LDPC BP 디코딩
- **PyTorch**: 이미지 prior 기반 EDM 디노이저

입출력 흐름은 다음과 같습니다.

1. Fashion-MNIST test 이미지(28x28)를 비트열(6272 bits)로 변환
2. CRC24A 추가 후 5G LDPC 인코딩 (`N=12600`)
3. BPSK(PAM-1) 변조 후 AWGN 채널 통과
4. APP demapper로 채널 LLR 생성
5. `LDPC5GDecoder_soft`가 BP 청크 반복 + 디노이저 extrinsic 결합
6. CRC 기반 BLER, 비트 비교 기반 BER 계산

---

## 2) 핵심 업데이트 식 (Turbo-style)

청크별 BP 이후(마지막 청크 제외) 아래를 계산합니다.

```text
bp_ext  = BP_post - payload_intr
src_ext = src_post - BP_post

new_input = channel + beta * bp_ext + alpha_t * src_ext
```

- `channel`: 초기 채널 LLR (`payload0`, 고정)
- `payload_intr`: 해당 청크 BP에 실제 입력된 a priori LLR
- `BP_post`: 해당 청크 BP posterior
- `src_post`: 디노이저 posterior logits
- `alpha_t`: 청크별 alpha schedule 값
- `beta`: BP extrinsic 가중치

즉, **BP extrinsic과 source extrinsic을 분리해 동시 반영**하는 구조입니다.

---

## 3) 저장소 구조 (핵심 파일)

- `decoder.py`: `LDPC5GDecoder_soft` 구현 (BP + denoiser 터보 루프)
- `denoiser.py`: TF↔PyTorch 브리지 (`SoftDenoiser`)
- `source_prior.py`: LLR↔이미지 변환 + EDM posterior + source extrinsic
- `score_denoiser/networks.py`: EDM SongUNet 백본
- `train_denoiser.py`: 디노이저 학습, `checkpoints/denoiser.pt` 생성
- `plot_comparison.py`: baseline vs proposed Eb/N0 곡선
- `plot_comparison_poster.py`: 포스터용 BLER/BER 분리 플롯
- `vanilla_sweep.py`: beta/alpha schedule/EbN0 스윕, CSV 누적
- `safe_sweep.sh`: 장시간 스윕 배치 실행 스크립트
- `sigma_sweep.py`: sigma 스윕
- `analyze_warmup.py`: warm-up 스윕 결과 분석
- `visualize_progression.py`, `visualize_progression_clean.py`: 디코딩 단계 시각화
- `stats_bler_ebno08.py`: Eb/N0=0.8 BLER 통계 + 신뢰구간 산출

---

## 4) 빠른 실행 가이드

작업 디렉토리:

```bash
cd /home/LJH/onlyextrinsic_vanilla
```

### 4.1 디노이저 학습(선택)

```bash
python train_denoiser.py --epochs 5 --batch 128 --lr 2e-4
```

출력: `checkpoints/denoiser.pt`

### 4.2 메인 비교 플롯

```bash
CUDA_VISIBLE_DEVICES=0 python plot_comparison_poster.py \
  --bp-schedule 5 5 5 5 5 5 \
  --alpha-schedule "0.10,0.08,0.06,0.04,0.02" \
  --beta 0.1 --sigma 0.3 \
  --ebno 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 \
  --baselines 30 --batch 64 --rounds 5 \
  --out results/comparison_5x6_beta01.png
```

출력:
- `results/comparison_5x6_beta01_bler.png`
- `results/comparison_5x6_beta01_ber.png`

### 4.3 Eb/N0=0.8 BLER 통계(현재 best 검증)

```bash
CUDA_VISIBLE_DEVICES=0 python stats_bler_ebno08.py \
  --ebno 0.8 --batch 64 --rounds 20
```

출력:
- `results/bler_stats_ebno08.json`
- `results/bler_stats_ebno08.csv`

---

## 5) 현재 Best 검증 결과 (인용용)

아래는 **실제로 최근 실행한 `stats_bler_ebno08.py` 결과**입니다.

조건:
- Eb/N0 = 0.8 dB
- Proposed schedule: `[5,5,5,5,5,5]`
- alpha schedule: `[0.10,0.08,0.06,0.04,0.02]`
- beta = 0.1, sigma = 0.3
- batch = 64, rounds = 20 (총 1280 blocks/decoder)

| Decoder | ACK / NACK | Pooled BLER | 95% CI (Wilson) | Round BLER mean±std | Pooled BER |
|---|---:|---:|---:|---:|---:|
| BP-30 (Baseline) | 632 / 648 | 0.506250 | [0.478883, 0.533580] | 0.506250 ± 0.061128 | 4.7743e-03 |
| Proposed [5x6] | 1276 / 4 | **0.003125** | [0.001216, 0.008008] | 0.003125 ± 0.008174 | 0.0000e+00 |

요약:
- Eb/N0=0.8 dB에서 Proposed는 Baseline 대비 BLER이 약 **160배** 낮음
- 본 저장소 기준 **현재 best 검증 수치**는 `Proposed BLER = 0.003125`

---

## 6) 결과 파일 형식

### `results/vanilla_sweep.csv` 컬럼

```text
timestamp,tag,mode,ebno_db,sigma,beta,alpha_schedule,
bler,ber,n_blocks,n_acks,n_nacks,n_bit_errors,bp_schedule
```

### `results/bler_stats_ebno08.json`

- `config`: 실행 파라미터
- `results`: 디코더별 BLER/BER/CI/라운드 통계

---

## 7) 의존성

- Python 3.8+
- TensorFlow 2.x
- PyTorch
- Sionna
- torchvision
- numpy, matplotlib

---

## 8) 주의사항

- `checkpoints/denoiser.pt`가 없으면 디노이저가 랜덤 가중치로 동작합니다.
- 기본 BLER/BER는 `batch * rounds` 샘플 수에 의존하므로, 논문급 통계를 위해 `rounds`를 충분히 키우는 것을 권장합니다.
- 현재 파이프라인은 TF↔PyTorch 브리지를 사용하므로 성능 최적화 여지가 있습니다.
