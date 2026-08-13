# MOHIM: Genius seed → YouTube → Demucs → ACE-Step LoRA

Genius에서 만든 가사 seed JSON과 YouTube 음원을 연결하고, HTDemucs 6-stem 분리와
반복 모티프 추출을 거쳐 ACE-Step 1.5 dual-stream LoRA 데이터를 만드는 연구
파이프라인이다.

## 1. 로컬 Mac에서 가사 seed 생성

Genius 수집은 Colab 노트북과 분리되어 있다. 로컬에서 다음 명령을 실행한다.

```bash
python3 -m pip install -r requirements.txt
python3 export_genius_seed.py --max-tracks 100 --max-pages 50
```

토큰은 프롬프트에 입력하거나 `GENIUS_ACCESS_TOKEN` 환경 변수로 전달한다. 기본 출력은
저장소 루트의 `genius_pop_seed.json`이다. 곡마다 즉시 저장되므로 중간에 중단해도 완료된
가사는 남는다. 이 단계에서는 YouTube 검색이나 음원 다운로드를 하지 않는다.

생성한 파일을 Google Drive의 다음 위치에 올린다.

```text
MyDrive/MOHIM/genius_pop_dataset/genius_pop_seed.json
```

## 2. Colab에서 음원과 학습 데이터 생성

[MOHIM_LoRA.ipynb](MOHIM_LoRA.ipynb)을 Colab에서 열고 위에서부터 실행한다.

```text
genius_pop_seed.json
→ YouTube 후보 검색 및 음원 다운로드
→ HTDemucs 6-stem 분리
→ 반복 4마디 모티프 선택
→ vocals를 제외한 stem을 accompaniment로 합산
→ dual_stream_manifest.json 생성
→ ACE-Step tensor 전처리
→ LoRA 학습
```

다운로드 결과는 다음처럼 Drive에 저장된다.

```text
MyDrive/MOHIM/genius_pop_dataset/
├── genius_pop_seed.json
├── tracks.json
└── audio/
```

`tracks.json`은 곡마다 갱신된다. unavailable 영상은 다음 검색 후보로 넘어가고, 이미 받은
음원은 재실행 시 건너뛴다.

Demucs 처리 결과는 다음처럼 저장된다.

```text
MyDrive/MOHIM/motif_dataset/<TRACK_ID>/
├── vocals.flac
├── accompaniment.flac  # vocals를 제외한 Demucs stem 전체 합
├── motif.flac           # 선택된 stem에서 추출한 4마디 조건
├── bass.flac            # Demucs가 반환한 경우 저장
├── guitar.flac
├── piano.flac
├── other.flac
├── lyrics.txt
└── metadata.json
```

모티프 선택 결과를 직접 비교할 수 있도록 vocals와 drums를 제외한 개별 stem도 저장한다.
drums는 개별 파일로 저장하지 않지만 `accompaniment.flac` 합산에는 포함된다.

처음에는 노트북의 `MAX_SONGS = 3`으로 결과를 듣고 확인한 뒤 전체 처리 시 `None`으로
바꾼다. 완성된 샘플은 다시 실행해도 건너뛴다.

## 3. ACE-Step

노트북은 공식 `ace-step/ACE-Step-1.5`를 호환 커밋
`6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`으로 고정하고
`patches/ace-step-1.5-dual-stream.patch`를 적용한다.

## 사용 조건

Genius 가사와 YouTube 음원의 이용 권한은 별도로 확인해야 한다. 연구 목적 자체가 복제나
다운로드 권한을 자동으로 부여하지 않으므로, 사용자가 접근·저장·학습할 권한이 있는 자료만
사용한다.

## 테스트

```bash
python -m unittest discover -s tests -v
```
