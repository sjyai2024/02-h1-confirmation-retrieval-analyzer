# H1 독립 확인표본(Confirmation Set) 프로토콜 v1
**고정일:** 2026-09-27  
**목적:** 기존 31개 개발표본에서 방법을 탐색·선정한 뒤, RQ1 결과를 보지 않은 A등급 로고 21개를 별도 확인표본으로 사용하여 H1을 단 한 번 검정한다.

## 1. 표본 분리
- 개발표본: 기존 Text–Visual 공통 A 브랜드 31개
- 확인표본: 기존 A등급 로고 중 텍스트 분석에 사용되지 않았던 **21개 브랜드**
- 확인표본의 CLIP wordmark 42-trait 결과는 새 텍스트 수집 전에 동결함.

### 동결 파일
- Seed CSV SHA256: `30b4261f5df5fbf7d0fe6f772a925ac32f166de32e5b5d10826bd89cf1673e5f`
- Frozen CLIP 42-trait SHA256: `60f703e205c9057143c830bb14caafeea56a0d07bab717fea5f37f293cd57f55`

## 2. 확인표본 21개
THE SAEM, CNP Laboratory, peripera, THE WHOO, DOMINAS, SOME BY MI, STEMBELL, medicube, fromrier, AMUSE, numbuzin, celimax, VDL, TIELA, farmrx, Dr.Jart+, LABO-H, ISA KNOX, beplain, CLIO, TPSY

## 3. 텍스트 수집
기존 01과 동일한 Collection-first 원칙을 사용한다.

### 자료원
- 공식 브랜드 웹사이트
- About / Brand / Story / Philosophy / Mission / Vision / Values / Heritage / Identity / Purpose
- 브랜드가 공식적으로 제공한 한국어 또는 영어 원문

### 제외
- 제품 상세 판매문구만 있는 페이지
- 리뷰, 뉴스, 이벤트, 로그인/장바구니/약관
- 브랜드 전략·정체성과 관계없는 UI 문구
- 다른 브랜드의 텍스트
- 연구자가 새로 작성한 요약·번역·의역

### 중요
- **CLIP 결과를 보면서 텍스트 포함/제외를 변경하지 않는다.**
- 수집 실패·텍스트 부족도 그대로 기록한다.
- 동일한 규칙으로 충분한 전략 텍스트를 확보하지 못한 브랜드는 사전 기준에 따라 `Insufficient_Text`로 기록한다.

## 4. 전처리
개발표본과 동일:
1. Unicode NFKC + whitespace normalization
2. sentence/proposition unit
3. 번호목록은 독립명제로 분할
4. 단일 heading은 인접문장에 결합
5. 의미가 독립적인 2단어 이상 slogan 유지
6. 연구자 paraphrase/translation/stemming/stopword removal 금지
7. normalized exact duplicate만 제거
8. 비명제 label-list 제외

최종 입력 필수열:
`Brand, Sentence_Unit_ID, Language, Sentence_Text`

## 5. 고정 텍스트 모델
**Primary:** `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`

개발표본에서 방법선정에 사용했음을 명시한다.
확인표본에서는 모델·prompt·score를 변경하지 않는다.

- Hypothesis: `This brand is {trait}.`
- Aaker 42 traits
- score: `P(entailment)`
- 한국어 원문은 번역하지 않는다.
- 브랜드별 trait score = 해당 브랜드 sentence/proposition의 entailment probability 평균

## 6. 고정 시각 모델
- 모델: generic CLIP wordmark 결과(03C)
- 입력: 기존 표준화 1024×1024 흰 배경·검정 영문 wordmark
- Aaker 42 traits
- prompt: `{trait} font`
- **확인표본 21개의 visual score는 이미 동결되어 다시 튜닝하지 않는다.**

## 7. H1
> **H1. 공식 브랜드 텍스트에서 도출한 브랜드 개성 프로파일은 비일치 브랜드의 영문 로고타입보다 동일 브랜드의 영문 로고타입과 더 높은 대응성을 보일 것이다.**

## 8. Primary confirmatory test
각 텍스트 브랜드 i와 모든 확인표본 wordmark j의 **42-trait Pearson profile correlation**을 계산한다.

각 브랜드 텍스트에서 자기 wordmark의 검색순위를 구한다.

### Primary statistic
**Mean Retrieval Rank**

대립가설:
`자기 wordmark의 평균순위 < 브랜드 라벨 무작위 치환 평균순위`

- permutation: 100,000회
- one-sided α=.05
- seed=20260927
- random expected mean rank = `(N+1)/2`

## 9. Secondary statistics
- Mean Reciprocal Rank (MRR)
- Recall@1
- Recall@3
- Recall@5
- matched mean profile r vs mismatched mean profile r
- 브랜드별 retrieval rank

Secondary 결과는 primary와 구분해 보고한다.

## 10. Confirmatory status
- 분석 가능한 확인브랜드가 **15개 이상**이면 confirmatory test로 실행.
- 15개 미만이면 결과는 `under-sized confirmation`으로 표시하고, 동일 기준의 신규 A-eligible K-cosmetic 브랜드를 추가 모집한다.
- p<.05 여부와 관계없이 결과를 그대로 보고한다.
- 확인결과를 본 뒤 모델·prompt·metric을 변경하지 않는다.

## 11. 개발표본과 확인표본의 역할
- 31개 개발표본: 방법개발·모델비교·탐색
- 21개 확인표본: 최종 H1 확인

따라서 개발표본에서 발견된 retrieval p값은 최종 가설검정으로 사용하지 않고,
확인표본 결과만 confirmatory H1의 판단근거로 사용한다.
