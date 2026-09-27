from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

APP_VERSION="1.2"
APP_DIR=Path(__file__).resolve().parent
MODEL_NAME="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
MAPPING_FILE=APP_DIR/"Aaker42_NLI_hypothesis_mapping.csv"

PRIMARY_PERMUTATIONS=100000
RANDOM_SEED=20260927

st.set_page_config(page_title="02C Uploaded Sample Retrieval",layout="wide")
st.title("02C · 업로드 표본 Text–Visual Retrieval 분석")
st.caption("승인 CSV의 브랜드를 기준으로 텍스트와 CLIP 로고 프로파일의 대응성을 분석합니다.")
st.info("표본 수는 업로드 파일에서 계산합니다. 기존 개발·탐색 데이터를 사용한 결과는 독립 확인표본 검정으로 해석하지 않습니다.")

@st.cache_data
def load_mapping():
    return pd.read_csv(MAPPING_FILE).sort_values("Item_No").reset_index(drop=True)

def bkey(s):
    s=str(s).strip().casefold().replace("ö","o")
    return re.sub(r"[^0-9a-z가-힣]+","",s)

def infer_language(text):
    """Script-based auxiliary metadata; never used to select or score text."""
    has_korean=bool(re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣᄀ-ᇿ]",str(text)))
    has_latin=bool(re.search(r"[A-Za-z]",str(text)))
    if has_korean and has_latin:
        return "mixed"
    if has_korean:
        return "ko"
    if has_latin:
        return "en"
    return "und"

def prepare_corpus(data):
    """Accept current/legacy columns without splitting or deduplicating units."""
    corpus=data.copy()
    corpus.columns=[str(col).replace("\ufeff","").strip() for col in corpus.columns]
    if corpus.columns.duplicated().any():
        raise ValueError("같은 이름의 열이 중복되어 있습니다. CSV 열 이름을 확인하세요.")

    info={"Input_Units":len(corpus),"Legacy_Columns_Mapped":{},"Excluded_Unapproved_Units":0}
    for old,new in (("Sentence_Unit_ID","Unit_ID"),("Sentence_Text","Text")):
        if old not in corpus.columns:
            continue
        if new in corpus.columns:
            current=corpus[new].fillna("").astype(str)
            legacy=corpus[old].fillna("").astype(str)
            conflict=(current.str.strip().ne("") & legacy.str.strip().ne("")
                      & current.str.strip().ne(legacy.str.strip()))
            if conflict.any():
                raise ValueError(f"{new}와 {old} 값이 서로 다른 행이 있습니다. 사용할 값을 확인하세요.")
            corpus[new]=current.where(current.str.strip().ne(""),legacy)
            corpus=corpus.drop(columns=[old])
        else:
            corpus=corpus.rename(columns={old:new})
        info["Legacy_Columns_Mapped"][old]=new

    missing={"Brand","Unit_ID","Text"}-set(corpus.columns)
    if missing:
        raise ValueError(f"필수 열 누락: {sorted(missing)}. 필수열은 Brand, Unit_ID, Text입니다.")

    # Preserve the researcher's existing decisions; do not reclassify content.
    if "Researcher_Final" in corpus.columns:
        decisions=corpus["Researcher_Final"].fillna("").astype(str).str.strip()
        numeric=pd.to_numeric(decisions,errors="coerce")
        invalid=decisions.ne("") & ~numeric.isin([0,1])
        if invalid.any():
            raise ValueError("Researcher_Final에는 0, 1 또는 빈칸만 사용할 수 있습니다.")
        approved=numeric.eq(1)
        info["Excluded_Unapproved_Units"]=int((~approved).sum())
        corpus=corpus.loc[approved].copy()

    if corpus.empty:
        raise ValueError("분석할 Content Unit이 없습니다. 승인 데이터와 Researcher_Final 값을 확인하세요.")
    for column in ("Brand","Unit_ID","Text"):
        values=corpus[column].fillna("").astype(str)
        if values.str.strip().eq("").any():
            raise ValueError(f"{column}에 빈 값이 있습니다. 분석할 Content Unit의 값을 확인하세요.")
        corpus[column]=values if column=="Text" else values.str.strip()

    unit_keys=pd.DataFrame({"Brand_Key":corpus["Brand"].map(bkey),"Unit_ID":corpus["Unit_ID"]})
    if unit_keys.duplicated().any():
        raise ValueError("같은 브랜드 안에서 Unit_ID가 중복됩니다. 승인 Unit의 식별자를 확인하세요.")

    if "Language" not in corpus.columns:
        corpus["Language"]=""
    language=corpus["Language"].fillna("").astype(str)
    blank_language=language.str.strip().eq("")
    corpus["Language"]=language
    corpus.loc[blank_language,"Language"]=corpus.loc[blank_language,"Text"].map(infer_language)
    info["Language_Inferred_Units"]=int(blank_language.sum())
    info["Approved_Units"]=len(corpus)
    return corpus.reset_index(drop=True),info

def read_corpus(source):
    # String loading preserves IDs such as 001 and literal text such as NA.
    return prepare_corpus(pd.read_csv(source,dtype=str,keep_default_na=False,encoding="utf-8-sig"))

def validate_visual(frame,trait_keys):
    frame=frame.copy()
    required=["Brand"]+["Raw_"+key for key in trait_keys]
    missing=set(required)-set(frame.columns)
    if missing:
        raise ValueError("시각 프로파일 필수열 누락: "+", ".join(sorted(missing)))
    if frame.empty or frame["Brand"].isna().any() or frame["Brand"].astype(str).str.strip().eq("").any():
        raise ValueError("시각 결과의 브랜드명이 비어 있습니다.")
    if frame["Brand"].map(bkey).duplicated().any():
        raise ValueError("시각 결과에 같은 브랜드가 중복됩니다. 브랜드별 프로파일 한 행이 필요합니다.")
    frame[required[1:]]=frame[required[1:]].apply(pd.to_numeric,errors="raise")
    if not np.isfinite(frame[required[1:]].to_numpy(float)).all():
        raise ValueError("시각 점수에 결측 또는 무한값이 있습니다.")
    return frame

def resolve_labels(model):
    out={}
    for i,v in model.config.id2label.items():
        lab=str(v).lower()
        if "entail" in lab: out["entailment"]=int(i)
        elif "neutral" in lab: out["neutral"]=int(i)
        elif "contrad" in lab: out["contradiction"]=int(i)
    if len(out)<3 and model.config.num_labels==3:
        out={"contradiction":0,"neutral":1,"entailment":2}
    if "entailment" not in out:
        raise ValueError(f"Cannot resolve entailment label: {model.config.id2label}")
    return out

@st.cache_resource(show_spinner=False)
def load_model():
    tok=AutoTokenizer.from_pretrained(MODEL_NAME)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    if torch.cuda.is_available():
        device=torch.device("cuda")
    elif getattr(torch.backends,"mps",None) and torch.backends.mps.is_available():
        device=torch.device("mps")
    else:
        device=torch.device("cpu")
    model.to(device).eval()
    return tok,model,device,resolve_labels(model)

def nli_entailment(tok,model,device,labels,premises,hypotheses,batch=16,max_length=256):
    vals=[]
    for s in range(0,len(premises),batch):
        enc=tok(
            premises[s:s+batch],hypotheses[s:s+batch],
            padding=True,truncation=True,max_length=max_length,return_tensors="pt"
        )
        enc={k:v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            probs=torch.softmax(model(**enc).logits,dim=-1)
        vals.extend(probs[:,labels["entailment"]].detach().cpu().numpy().tolist())
    return np.asarray(vals,float)

def profile_corr(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if np.std(a)<1e-12 or np.std(b)<1e-12: return np.nan
    return float(np.corrcoef(a,b)[0,1])

def similarity_matrix(T,V):
    M=np.empty((len(T),len(V)),float)
    for i in range(len(T)):
        for j in range(len(V)):
            M[i,j]=profile_corr(T[i],V[j])
    return M

def retrieval_ranks(M):
    ranks=[]
    for i in range(len(M)):
        order=np.argsort(-M[i],kind="stable")
        ranks.append(int(np.where(order==i)[0][0])+1)
    return np.asarray(ranks,int)

def permutation_mean_rank(M,n_perm=PRIMARY_PERMUTATIONS,seed=RANDOM_SEED):
    n=M.shape[0]
    actual=retrieval_ranks(M)
    observed=float(actual.mean())
    rng=np.random.default_rng(seed)
    vals=np.empty(n_perm,float)
    all_orders=[np.argsort(-M[i],kind="stable") for i in range(n)]
    inv_ranks=[]
    for order in all_orders:
        inv=np.empty(n,int)
        inv[order]=np.arange(1,n+1)
        inv_ranks.append(inv)
    inv_ranks=np.vstack(inv_ranks)
    for k in range(n_perm):
        p=rng.permutation(n)
        vals[k]=np.mean([inv_ranks[i,p[i]] for i in range(n)])
    pval=(np.sum(vals<=observed)+1)/(n_perm+1)
    return actual,observed,float(vals.mean()),float(pval)

def permutation_matched_r(M,n_perm=PRIMARY_PERMUTATIONS,seed=RANDOM_SEED):
    n=M.shape[0]
    obs=float(np.nanmean(np.diag(M)))
    off=float(np.nanmean(M[~np.eye(n,dtype=bool)]))
    rng=np.random.default_rng(seed)
    vals=np.empty(n_perm,float)
    for k in range(n_perm):
        p=rng.permutation(n)
        vals[k]=np.nanmean([M[i,p[i]] for i in range(n)])
    pval=(np.sum(vals>=obs)+1)/(n_perm+1)
    return obs,off,float(pval)

def zip_outputs(files,meta):
    bio=io.BytesIO()
    with zipfile.ZipFile(bio,"w",zipfile.ZIP_DEFLATED) as z:
        for name,df in files.items():
            z.writestr(name,df.to_csv(index=False).encode("utf-8-sig"))
        z.writestr("02C_99_metadata.json",json.dumps(meta,ensure_ascii=False,indent=2))
    return bio.getvalue()

if not MAPPING_FILE.is_file():
    st.error("app.py와 같은 폴더에 Aaker42_NLI_hypothesis_mapping.csv가 필요합니다.")
    st.stop()
mapping=load_mapping()
trait_keys=mapping["Trait_Key"].tolist()
if len(trait_keys)!=42 or len(set(trait_keys))!=42:
    st.error("매핑 파일에는 중복 없이 42개 Trait_Key가 필요합니다.")
    st.stop()
st.markdown("### 분석규칙")
st.code("Text: mDeBERTa NLI · This brand is {trait}. · P(entailment)\nVisual: CLIP raw 42-trait\nSimilarity: Pearson r · Mean Retrieval Rank\nPermutation: 100,000 one-sided · alpha=.05\nN: 업로드 텍스트와 시각 결과의 공통 브랜드 수")
visual_upload=st.file_uploader("전체 브랜드 CLIP raw 42-trait 결과 CSV 또는 ZIP",type=["csv","zip"],key="visual")
visual=None
visual_source=""
if visual_upload is not None:
    try:
        if visual_upload.name.lower().endswith(".zip"):
            with zipfile.ZipFile(visual_upload) as archive:
                candidates={}
                for name in archive.namelist():
                    if name.lower().endswith(".csv"):
                        frame=pd.read_csv(io.BytesIO(archive.read(name)))
                        if {"Brand",*("Raw_"+key for key in trait_keys)}.issubset(frame.columns):
                            candidates[name]=frame
                if not candidates:
                    raise ValueError("ZIP에 Brand 및 Raw_<Trait_Key> 42개 열을 가진 CSV가 없습니다. 원점수 프로파일 CSV를 사용하세요.")
                chosen=st.selectbox("시각 프로파일 파일",list(candidates))
                visual=candidates[chosen]
                visual_source=visual_upload.name+"/"+chosen
        else:
            visual=pd.read_csv(visual_upload)
            visual_source=visual_upload.name
        visual=validate_visual(visual,trait_keys)
        if "Eligibility" in visual.columns:
            quality_scope=st.selectbox("로고 등급 범위",["모든 등급 (탐색용)","A등급만 (기존 기준)"])
            if quality_scope.startswith("A등급"):
                visual=visual.loc[visual["Eligibility"].astype(str).str.strip().str.upper().eq("A")].copy()
        else:
            quality_scope="등급 정보 없음"
            st.warning("시각 파일에 Eligibility가 없어 로고 등급을 확인할 수 없습니다.")
    except (ValueError,UnicodeError,zipfile.BadZipFile,pd.errors.ParserError) as exc:
        st.error(f"시각 결과 입력 확인: {exc}")
        st.stop()
else:
    st.caption("기존 21개 전용 시각 파일은 자동 사용하지 않습니다. 업로드 텍스트의 브랜드를 포함한 전체 CLIP 결과를 선택하세요.")

uploaded=st.file_uploader(
    "최종 승인 Content Unit CSV 업로드",
    type=["csv"],
    help=("필수열: Brand, Unit_ID, Text. 이전 Sentence_Unit_ID/Sentence_Text 형식도 지원합니다. "
          "Language는 없거나 비어 있으면 자동 생성합니다. "
          "Researcher_Final 열이 있으면 1인 행만 사용합니다.")
)

if uploaded is not None:
    try:
        corpus,input_info=read_corpus(uploaded)
    except (ValueError,UnicodeError,pd.errors.ParserError) as exc:
        st.error(f"CSV 입력 확인: {exc}")
        st.stop()

    if input_info["Legacy_Columns_Mapped"]:
        st.info("이전 열 이름을 Unit_ID·Text로 변환했습니다. Content Unit은 다시 분할하지 않습니다.")
    if input_info["Excluded_Unapproved_Units"]:
        st.info(f"Researcher_Final이 1인 행만 사용합니다. 미승인 {input_info['Excluded_Unapproved_Units']}개 Unit을 제외했습니다.")
    if input_info["Language_Inferred_Units"]:
        st.caption("빈 Language 값은 문자 구성으로 추정했습니다(ko/en/mixed/und). 분석 점수에는 사용하지 않습니다.")

    corpus["_key"]=corpus["Brand"].map(bkey)
    brand_names=corpus.drop_duplicates("_key").set_index("_key")["Brand"].to_dict()
    corpus["Brand"]=corpus["_key"].map(brand_names)
    counts=corpus.groupby("Brand").size().rename("N_Units").reset_index()
    n_input=len(brand_names)
    st.success(f"입력 표본: {n_input}개 브랜드 · 승인 Content Unit {len(corpus)}개")
    st.dataframe(counts,use_container_width=True,hide_index=True)
    if visual is None:
        st.info("텍스트 CSV를 읽었습니다. 비교할 전체 브랜드 CLIP 결과도 업로드하세요.")
        st.stop()
    visual["_key"]=visual["Brand"].map(bkey)
    common_keys=set(corpus["_key"]) & set(visual["_key"])
    coverage=counts.copy()
    coverage["Visual_Available"]=coverage["Brand"].map(bkey).isin(common_keys)
    st.dataframe(coverage,use_container_width=True,hide_index=True)
    st.info(f"텍스트 {n_input}개 / 시각 결과와 공통 {len(common_keys)}개 / 시각 결과 누락 {n_input-len(common_keys)}개")
    if len(common_keys)<2:
        st.error("브랜드 간 비교에는 공통 브랜드가 최소 2개 필요합니다. 이는 검정력 기준이 아닌 계산상의 최소 조건입니다.")
        st.stop()
    if len(common_keys)<n_input:
        st.warning("시각 결과가 없는 브랜드는 텍스트 분석에는 포함되지만 Text–Visual 비교에서는 제외됩니다.")
    if st.button("업로드 표본 분석 실행",type="primary"):
        tok,model,device,labels=load_model()
        trait_keys=mapping["Trait_Key"].tolist()
        hypotheses=[f"This brand is {t}." for t in mapping["Trait"]]

        rows=[]
        total=len(corpus)*len(mapping)
        prog=st.progress(0.0,text=f"NLI {total:,} pairs")
        done=0
        # Process one trait at a time for deterministic memory use.
        for j,(_,a) in enumerate(mapping.iterrows()):
            premises=corpus["Text"].tolist()
            hyps=[hypotheses[j]]*len(premises)
            vals=nli_entailment(tok,model,device,labels,premises,hyps)
            for idx,val in enumerate(vals):
                rows.append({
                    "Brand":corpus.iloc[idx]["Brand"],
                    "Unit_ID":corpus.iloc[idx]["Unit_ID"],
                    "Language":corpus.iloc[idx]["Language"],
                    "Trait":a["Trait"],
                    "Trait_Key":a["Trait_Key"],
                    "Entailment_Prob":float(val),
                })
            done += len(premises)
            prog.progress(done/total,text=f"NLI {done:,}/{total:,}")
        prog.empty()

        unit_trait=pd.DataFrame(rows)
        brand_trait=(unit_trait.groupby(["Brand","Trait_Key"],as_index=False)["Entailment_Prob"].mean()
                     .pivot(index="Brand",columns="Trait_Key",values="Entailment_Prob").reset_index())

        # Align uploaded visual profiles by brand key.
        T=brand_trait.copy(); V=visual.copy()
        T["_key"]=T["Brand"].map(bkey); V["_key"]=V["Brand"].map(bkey)
        common=sorted(set(T["_key"])&set(V["_key"]))
        T=T.set_index("_key").loc[common].reset_index()
        V=V.set_index("_key").loc[common].reset_index()

        status="UPLOADED_SAMPLE_EXPLORATORY"

        text_cols=trait_keys
        visual_cols=["Raw_"+x for x in trait_keys]
        A=T[text_cols].to_numpy(float)
        B=V[visual_cols].to_numpy(float)
        if not np.isfinite(A).all() or not np.isfinite(B).all() or np.any(np.std(A,axis=1)<1e-12) or np.any(np.std(B,axis=1)<1e-12):
            st.error("공통 브랜드에 결측·무한값 또는 분산이 0인 프로파일이 있어 Pearson 비교를 중단했습니다.")
            st.stop()
        M=similarity_matrix(A,B)

        ranks,mean_rank,perm_mean,p_rank=permutation_mean_rank(M)
        matched_r,nonmatched_r,p_r=permutation_matched_r(M)

        detail=pd.DataFrame({
            "Brand_Text":T["Brand"],
            "Brand_Visual":V["Brand"],
            "Matched_Profile_r":np.diag(M),
            "Retrieval_Rank":ranks,
            "Reciprocal_Rank":1/ranks,
            "Top1":(ranks<=1).astype(int),
            "Top3":(ranks<=3).astype(int),
            "Top5":(ranks<=5).astype(int),
        })
        n=len(detail)
        summary=pd.DataFrame([{
            "Status":status,
            "N_Input_Brands":n_input,
            "N_Matched_Brands":n,
            "N_Units":len(corpus),
            "Mean_Retrieval_Rank":mean_rank,
            "Random_Expected_Mean_Rank":(n+1)/2,
            "Permutation_N":PRIMARY_PERMUTATIONS,
            "Primary_OneSided_p":p_rank,
            "MRR":detail["Reciprocal_Rank"].mean(),
            "Recall_at_1":detail["Top1"].mean(),
            "Recall_at_3":detail["Top3"].mean(),
            "Recall_at_5":detail["Top5"].mean(),
            "Matched_Mean_Profile_r":matched_r,
            "Nonmatching_Mean_Profile_r":nonmatched_r,
            "Secondary_Profile_r_p":p_r,
            "Exploratory_Primary_Significant":bool(p_rank<0.05 and mean_rank<(n+1)/2),
        }])

        sim_df=pd.DataFrame(M,index=T["Brand"],columns=V["Brand"]).reset_index(names="Text_Brand")

        st.header("업로드 표본 분석 결과")
        st.dataframe(summary,use_container_width=True,hide_index=True)
        st.dataframe(detail.sort_values("Retrieval_Rank"),use_container_width=True,hide_index=True)

        files={
            "02C_00_content_units.csv":corpus.drop(columns=["_key"]),
            "02C_01_unit_trait_scores.csv":unit_trait,
            "02C_02_brand_trait_profiles.csv":brand_trait,
            "02C_03_similarity_matrix.csv":sim_df,
            "02C_04_brand_retrieval_detail.csv":detail,
            "02C_05_uploaded_sample_summary.csv":summary,
            "02C_06_brand_coverage.csv":coverage,
        }
        meta={
            "App_Version":APP_VERSION,
            "Input_Schema":"Content Unit: Brand, Unit_ID, Text; Language optional",
            "Input_Processing":input_info,
            "Model":MODEL_NAME,
            "Hypothesis":"This brand is {trait}.",
            "Text_Score":"P(entailment)",
            "Visual_Profile":"uploaded CLIP wordmark raw 42-trait",
            "Visual_Source":visual_source,
            "Visual_Quality_Scope":quality_scope,
            "N_Input_Brands":n_input,
            "N_Matched_Brands":n,
            "Primary_Similarity":"Pearson r across raw 42-trait profiles",
            "Primary_Statistic":"Mean Retrieval Rank",
            "Permutation_N":PRIMARY_PERMUTATIONS,
            "Permutation_Seed":RANDOM_SEED,
            "Alpha":0.05,
            "Independent_Confirmation":False,
            "Status":status,
        }
        st.download_button(
            "업로드 표본 결과 ZIP 다운로드",
            data=zip_outputs(files,meta),
            file_name="02C_uploaded_sample_retrieval_results.zip",
            mime="application/zip"
        )
