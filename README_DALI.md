# DALI v2 → MOHIM LoRA

기존 `MOHIM.ipynb`를 변경하지 않고 DALI v2의 영어 Pop 곡에서 반복 모티프를 골라
MOHIM dual-stream LoRA 데이터를 만드는 Colab 파이프라인이다.

## 준비물

1. Zenodo에서 비상업 연구 목적으로 접근 승인을 받은 DALI v2 annotation 디렉터리
2. DALI ID를 파일명으로 사용하는 로컬 음원 디렉터리
3. NVIDIA CUDA 런타임과 충분한 Google Drive 공간

DALI는 원본 음원을 함께 배포하지 않는다. 이 프로젝트는 사용자가 적법하게 확보해 둔
로컬 음원만 처리하며 YouTube 자동 다운로드 기능을 제공하지 않는다.

## Colab 실행

저장소를 clone한 다음 `notebooks/MOHIM_DALI_v2_LoRA.ipynb`를 열고 셀을 위에서부터
실행한다. 첫 실행은 반드시 `MAX_SONGS = 3`으로 결과를 듣고 확인한다.

노트북의 설정 셀에서 다음 경로만 수정하면 된다.

```python
DALI_DATA_DIR = "/content/drive/MyDrive/MOHIM/dali_v2"
DALI_AUDIO_DIR = "/content/drive/MyDrive/MOHIM/dali_audio"
OUTPUT_DIR = "/content/drive/MyDrive/MOHIM/dali_motif_dataset"
```

## 저장 공간

DALI v2는 전체 488.1시간이다. 압축 원본이 128–256 kbps라면 약 28–56GB지만, 이를
44.1kHz stereo PCM-16 WAV로 풀면 스트림 하나가 약 310GB가 된다. 이 파이프라인은
Demucs의 여섯 스템을 모두 저장하지 않고 `vocals`, 선택된 전체 motif stem, 4마디
`motif`만 FLAC으로 저장한다. 그래도 전체 처리에는 수백 GB가 필요할 수 있으므로 먼저
작은 subset으로 필요한 실제 용량을 측정한다.

## 출력

```text
<OUTPUT_DIR>/<DALI_ID>/
├── vocals.flac
├── guitar.flac          # 곡에 따라 piano/bass/other
├── motif.flac
├── lyrics.txt
└── metadata.json
```

`build_report.jsonl`에는 채택 및 제외 사유가 기록되고, `dual_stream_manifest.json`은
MOHIM의 `train.py fixed --dual-stream` 전처리에 전달된다. 완료된 샘플은 다시 실행해도
건너뛴다.

## 테스트

```bash
python -m unittest discover -s tests -v
```
