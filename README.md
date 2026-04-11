# ToonPT: Multimodal Retrieval System for Webtoon Question Answering

> **웹툰 컷을 장면 단위로 이해하고, 사용자의 자연어 질문에 정확한 화/씬을 찾아 답변하는 Multimodal RAG 시스템**  
> _BOAZ 24기 분석 컨퍼런스 · TEAM 어벤정스_  
> _정명훈 · 고혜정 · 정예린 · 정현서_
<img src="assets/team.png">

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-red) ![EXAONE](https://img.shields.io/badge/LLM-EXAONE-orange) ![ko--sbert](https://img.shields.io/badge/embedding-ko--sbert-green)

> _"그때 그 장면 몇 화였지?"_ — 연재가 길어진 웹툰에서 독자가 특정 장면을 다시 찾기 위해 스크롤을 거슬러 올라가는 경험을, **자연어 질문 한 줄로 해결**하는 것이 ToonPT의 목표입니다.

📄 참조 논문: [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval (Sarthi et al., 2024)](https://arxiv.org/abs/2401.18059)

---

## 1. 문제 정의: 왜 웹툰 RAG가 어려운가

웹툰은 일반적인 문서 RAG와 근본적으로 다른 두 가지 특성이 있습니다.

| 일반 문서 | 웹툰 |
|---|---|
| 텍스트가 1차 정보 | **이미지가 1차 정보**, 텍스트(대사)는 보조 |
| 단락 단위로 자연 분할 | **세로 스크롤 형태**, 컷 경계가 명시되지 않음 |
| 의미가 한 단락 안에서 완결 | **연속된 컷의 흐름**으로 의미 형성 |

→ 단순 텍스트 임베딩으로는 부족하고, **이미지 + 대사 + 서사 흐름**을 모두 다루는 멀티모달 파이프라인이 필요합니다.

## 2. Architecture 개요

전체 시스템은 **인덱싱 파이프라인**(상)과 **질의 응답 파이프라인**(하) 두 단계로 구성됩니다. 핵심 설계는 두 가지:

- **DB를 두 종류로 분리**: 씬 단위는 **Vector DB(Qdrant)** 에 임베딩 저장, 화 요약·사건·전체 줄거리·characters 사전은 **Look-up DB**에 그대로 저장
- **Router가 질문 유형에 따라 경로 분기**: 단순 화 요약 질문은 Look-up만, 장면 찾기 질문은 Vector 검색 + Rerank 경유

<p align="center">
  <img src="assets/architecture.png" width="90%"/>
  <br/>
  <em>상: 컷 분할 → Caption + OCR → 계층적 서사 구조 → ko-sbert 임베딩 → Qdrant·Look-up DB / 하: User → Router → (Look-up | Query Expansion → Hybrid Search → Rerank) → EXAONE → Answer</em>
</p>

## 3. 데이터 전처리: Dynamic Cut Segmentation

웹툰 raw data는 한 화당 수천 픽셀의 세로 스크롤 이미지입니다. 이를 **컷 단위로 자동 분할**하는 것이 모든 후속 처리의 출발점입니다.

- **여백 비율 기반 경계 탐지**: 컷 사이의 흰색 여백 영역을 감지해 자동 분할
- **세로로 긴 컷 추가 분할**: 한 컷이 비정상적으로 길면 비율 기반으로 다시 쪼갬 (단일 컷에 여러 장면이 들어간 경우 대응)

<p align="center">
  <img src="assets/cut_segmentation.png" width="70%"/>
  <br/>
  <em>스크롤 raw → 컷 단위 분할. 긴 컷은 추가로 비율 분할되어 장면 단위 검색이 가능해짐</em>
</p>

## 4. 멀티모달 정보 추출: Caption × OCR 이중 트랙

웹툰의 한 컷에는 **시각적 맥락**(인물 표정·구도)과 **언어적 정보**(대사·효과음)가 공존합니다. 이를 한 모델로 한꺼번에 처리하면 한쪽이 다른 쪽에 묻히는 문제가 생겨, **두 트랙으로 분리 추출**했습니다.

### 4-1. Image Captioning (맥락 트랙)
- **JoyCaption**으로 컷의 시각 정보 추출
- **연속된 4컷을 하나의 context window**로 묶어 캡션 생성 → 단일 컷의 의미 모호성을 인접 컷의 맥락으로 보완

### 4-2. Text OCR (근거 트랙)
- **Comic-Text-Detector**로 말풍선 영역 탐지
- 탐지된 영역에 **Clova OCR** 적용하여 대사·효과음 추출

<p align="center">
  <img src="assets/caption_ocr_example.png" width="75%"/>
  <br/>
  <em>좌: Caption summary (시각 맥락) / 우: Text OCR (대사 근거) — 두 정보를 분리 추출 후 통합</em>
</p>

→ Caption은 *"이 장면에서 무슨 일이 일어나는가"*, OCR은 *"누가 무슨 말을 했는가"* 를 담당. 검색 시 두 정보가 상호 보완.

## 5. 상향식 서사 계층 구축 (RAPTOR-inspired)
웹툰은 **장기 의존성**이 강한 매체입니다. 특정 장면의 의미는 그 장면 단독이 아니라 앞 화의 사건, 작품 전체의 서사 안에서 결정됩니다. 이를 단일 임베딩 층으로 다루면 검색이 표면적 매칭에 머무는 문제가 있어, **추상화 수준이 다른 여러 층으로 정보를 쌓는** RAPTOR의 hierarchical summarization 아이디어를 차용했습니다.
<br>

Scene (4컷 윈도우, 임베딩 검색 대상)
↓ EXAONE 요약
Chapter (한 화 요약, Look-up)
↓ EXAONE 사건 추출
Event (사건 단위, Look-up)
↓ EXAONE 통합 요약
Full Synopsis (작품 전체 줄거리, Look-up)

Characters (인물 사전, Query Expansion·답변 보강용)
| 계층 | 단위 | 저장 위치 | 검색 시 활용 |
|---|---|---|---|
| **Scene** | 4컷 윈도우 | **Vector DB (Qdrant)** | 구체적 장면 질의 — 임베딩 검색 |
| **Chapter** | 한 화 | Look-up DB | 화 요약 질의 — 직접 조회 |
| **Event** | 사건 단위 | Look-up DB | 답변 생성 시 컨텍스트 보강 |
| **Full Synopsis** | 작품 전체 | Look-up DB | 답변 생성 시 컨텍스트 보강 |
| **Characters** | 인물 사전 | Look-up DB | Query Expansion + 답변 생성 |

→ **모든 계층을 Vector DB에 넣지 않는 이유**: 화 요약·전체 줄거리는 "검색"의 대상이 아니라 "참조"의 대상이기 때문. Router가 질문 의도를 파악해 필요한 계층만 직접 가져오는 편이 정확하고 빠릅니다.

## 6. 검색 파이프라인: Router로 두 갈래 분기

### 1️⃣ User Input
사용자의 자연어 질문 (예: *"맥스와 조이가 우산을 같이 쓴 장면은 몇 화인가요?"*)

### 2️⃣ LLM Router (EXAONE)
질문 의도를 분류해 **두 경로 중 하나**로 분기:
- **단순 Look-up 경로** — "3화 무슨 내용?" 같은 화 단위 질문 → 화 키워드 추출 → Look-up DB에서 화 요약·사건·전체 줄거리·characters 직접 조회 → 바로 답변 생성
- **검색 경로** — "우산을 같이 쓴 장면" 같은 장면 찾기 질문 → 아래 3~5단계 진행

### 3️⃣ Query Expander
원본 쿼리에 `characters.json`(인물 사전)을 결합해 쿼리 재작성. 별명·애칭·관계어 → 정식 인물명으로 매핑하여 **객체 불일치 문제**(예: "맥스" = "강동구") 해결.

### 4️⃣ Hybrid Search
- **Dense Retrieval**: ko-sbert (`jhgan/ko-sbert-nli`) 임베딩 → 코사인 유사도
- **Sparse Retrieval**: 쿼리에서 키워드 추출 → BM25
- **RRF (Reciprocal Rank Fusion)**: 두 결과의 score 단위가 다르므로 **순위 기반**으로 점수 합산 → Top-50 추출

> 💡 BM25를 함께 쓰는 이유: 인물명·고유명사처럼 lexical overlap이 결정적인 쿼리에서 dense retriever가 약한 문제를 보완.

### 5️⃣ Cross-encoder Reranker
BERT 기반 cross-encoder가 **50개 문서와 쿼리를 각각 페어로 비교**해 새로운 점수를 산출 → 최종 **Top-5** 선별. (Bi-encoder인 ko-sbert와 달리 cross-encoder는 쿼리·문서를 함께 인코딩해 더 정밀한 관련도 측정이 가능)

### 6️⃣ Final Generation
Top-5 씬 + characters 사전 + 전체 요약을 EXAONE에 프롬프트로 전달 → 최종 답변 생성. **Streamlit**으로 구현한 UI에서 답변과 참조 씬을 함께 보여줌.
### 5️⃣ Re-ranking & Filtering
- Scene + Chapter + Event + Full Synopsis 4개 계층에서 **Candidate 50개** 추출
- Reranker로 **Top-5** 최종 선별

### 6️⃣ Final Generation
EXAONE이 Top-5 컨텍스트를 받아 최종 답변 생성.

## 7. 평가

### 평가 데이터
- 대상: 네이버 웹툰 『이직로그』
- ⚖️ **저작권법 제35조의5(저작물의 공정한 이용)에 의거하여 연구 목적으로만 활용**
- 독자 perspective 질문 리스트 작성, 정답 라벨링 (정답 위치 + 난이도 1–5)

### 평가 지표
| 단계 | 지표 | 설명 |
|---|---|---|
| **Retrieval** | Recall@3 (Chapter), Recall@5 (Scene) | 정답 화/씬이 검색 결과에 포함되는가 |
| **Generation** | Exact Match (EM) | 모델 답변이 정답 개체명과 정확히 일치하는가 |

EM은 **검색 성공 ≠ 생성 성공**을 분리해서 보기 위함. 예: 정답 "맥스"인데 모델이 "엔디 님"이라 답하면 → 검색은 성공해도 생성 단계에서 hallucination 발생.

## 8. 결과

<p align="center">
  <img src="assets/chapter_accuracy.png" width="40%"/>
  <img src="assets/difficulty_accuracy.png" width="55%"/>
  <br/>
  <em>좌: 전체 chapter 식별 정확도 약 68% / 우: 난이도별 정확도 — 1~3단계 80%+, 5단계 ~40%</em>
</p>

### 핵심 수치
- **에피소드/씬 식별 정확도 약 68%** — 10번의 질문 중 약 7번은 정확한 문맥을 참조하여 답변
- **난이도 1–3** (단순 키워드/명확한 질문): **평균 80% 이상**
- **난이도 4–5** (모호한 질문/추상적 표현): **40–55%로 급격히 하락**

### 핵심 진단: Semantic Gap
난이도 4–5 구간의 성능 하락은 **텍스트 임베딩만으로는 해결하기 어려운 의미 격차(Semantic Gap)** 를 보여줍니다. 사용자가 *"두 사람이 어색하게 침묵하는 장면"* 처럼 묘사할 때, 캡션과 OCR에는 그 단어가 직접 등장하지 않아 검색이 실패하는 케이스.

### 정성 분석

| 케이스 | 질문 | 결과 |
|---|---|---|
| ✅ 성공 | "맥스와 조이가 우산을 같이 쓴 장면" | 14화 50–69컷 정확히 검색·답변 |
| ✅ 성공 | "에스더가 전 남자친구의 바람을 알아챈 장면" | 10화 32컷 정확 |
| ❌ 실패 | "조이가 맥스에게 벚꽃을 뿌려주는 장면" (정답 7화 후반) | 5화 63컷으로 오답 — 시각적 묘사 매칭 실패 |
| ❌ 실패 | "조이가 맥스를 이직스터디 파트너로 정한 이유" (정답 4화 후반) | 3화 77컷으로 오답 — 추론형 질문 |

## 9. Conclusion

1. **멀티모달 데이터 구조화** — 시각 요소(JoyCaption) + 대사(OCR) + 상세 설명을 통합해 LLM이 시각 매체인 웹툰을 텍스트 기반으로 이해할 수 있는 고품질 데이터셋 구축
2. **상향식 서사 계층** — Scene → Chapter → Synopsis → Event의 4단 계층으로 다양한 추상화 수준의 질의에 대응
3. **RAG 파이프라인 최적화** — Hybrid Search(Vector + BM25) + Re-ranking 도입으로 사용자의 모호한 질문에도 정확한 씬 검색

## 10. Future Direction

### 데이터 범용성 및 확장성
- 특정 웹툰의 설정(등장인물, 세계관)에 종속되지 않는 **컷 단위 공통 JSON 스키마** 설계
- 장르·연출 방식이 다른 웹툰에도 추가 규칙 정의 없이 데이터셋 확장 가능

### 신규 회차 자동 인덱싱 파이프라인 (B2B 상용화의 핵심)
ToonPT는 **B2B 모델**을 전제로 설계되었습니다. 즉 독자가 새 화 업데이트를 확인하는 용도가 아니라, **플랫폼이 신규 화를 올릴 때마다 백엔드에서 자동으로 인덱스를 갱신**하는 용도. 이를 위해:
- 신규 화 업데이트 시 전체 재처리 없이 **신규 컷만 자동 감지** → Caption · OCR · 메타데이터 생성
- 기존 Vector DB와 Look-up DB에 증분 추가 → 최신 내용이 검색에 즉시 반영

이 자동화 파이프라인이 완성되어야 진짜 상용 가능한 서비스가 됩니다.

### 모델 측면
- 웹툰 특화 연출 인식을 위한 **VLM 파인튜닝** (특히 난이도 4–5 구간 개선용)
- Multimodal embedding 직접 도입 (CLIP 계열) — 현재 텍스트화 후 임베딩하는 우회 방식의 한계 보완

---

## 🛠️ 기술 스택

**Vision & OCR**: `JoyCaption` · `Comic-Text-Detector` · `Clova OCR`  
**LLM**: `EXAONE` (Router · Summarization · Generation)  
**Embedding**: `ko-sbert (jhgan/ko-sbert-nli)`  
**Retrieval**: `Qdrant` (Vector DB) · `BM25` · `RRF` · `Reranker`  
**Language**: `Python`
**UI**: `Streamlit`

## 📚 참고 문헌

- Sarthi, P. et al. (2024). [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059). *ICLR 2024*.
- 본 프로젝트는 RAPTOR의 hierarchical summarization 아이디어를 웹툰의 서사 구조(Scene → Chapter → Synopsis → Event)에 맞게 재구성하여 적용했습니다.

## 📁 저장소 구조

```text
├── assets/                    # README 이미지
├── preprocessing/             # Dynamic Cut Segmentation
├── extraction/                # JoyCaption + Comic-Text-Detector + Clova OCR
├── indexing/                  # Scene → Chapter → Synopsis → Event 계층 생성
├── retrieval/                 # Router + Query Expander + Hybrid Search + Reranker
├── generation/                # EXAONE 답변 생성
├── evaluation/                # Recall@k + EM 평가
└── ToonPT_발표자료.pdf
```
