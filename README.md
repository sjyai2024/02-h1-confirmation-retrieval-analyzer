# 02C H1 Confirmation Retrieval Analyzer v1.0

## 목적
개발표본 31개에서 선택한 최종 계산방법을 결과 미확인 A등급 21개 브랜드에서 단 한 번 확인합니다.

## 먼저 할 일
1. `H1_confirmation_set_21_seed.csv`를 기존 01 Brand Text Collector v2.8에 업로드
2. Collection-first로 공식 텍스트 원자료 수집
3. 기존과 동일한 고정 inclusion/exclusion 규칙으로 수동 정제
4. 기존과 동일한 sentence/proposition 규칙으로 최종 CSV 작성
5. 필수열:
   - Brand
   - Sentence_Unit_ID
   - Language
   - Sentence_Text

## Confirmation 앱 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

최종 corpus를 업로드하고 `H1 확인분석 실행`.

앱에서 바꿀 수 없는 고정값:
- mDeBERTa
- Aaker English original 42 traits
- generic CLIP frozen wordmark 42 traits
- Pearson profile correlation
- Mean Retrieval Rank
- permutation 100,000
- alpha .05

분석가능 브랜드가 15개 미만이면 confirmatory 판정을 하지 않습니다.
