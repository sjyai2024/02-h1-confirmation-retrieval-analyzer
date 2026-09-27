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

APP_VERSION="1.0"
APP_DIR=Path(__file__).resolve().parent
MODEL_NAME="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
VISUAL_FILE=APP_DIR/"H1_confirmation_21_CLIP_42trait_FROZEN.csv"
SEED_FILE=APP_DIR/"H1_confirmation_set_21_seed.csv"
MAPPING_FILE=APP_DIR/"Aaker42_NLI_hypothesis_mapping.csv"

PRIMARY_PERMUTATIONS=100000
RANDOM_SEED=20260927

st.set_page_config(page_title="02C H1 Confirmation Retrieval",layout="wide")
st.title("02C · H1 독립 확인표본 Retrieval Test")
st.caption("개발표본 31개와 분리된 A등급 21개 브랜드에서 mDeBERTa Text 42-trait ↔ frozen CLIP wordmark 42-trait를 단 한 번 확인")

st.warning(
    "이 앱은 **확인표본(confirmatory set)** 전용입니다. "
    "모델, prompt, visual profile, primary metric은 고정되어 있으며 화면에서 선택할 수 없습니다."
)

@st.cache_data
def load_seed():
    return pd.read_csv(SEED_FILE)

@st.cache_data
def load_visual():
    return pd.read_csv(VISUAL_FILE)

@st.cache_data
def load_mapping():
    return pd.read_csv(MAPPING_FILE).sort_values("Item_No").reset_index(drop=True)

def bkey(s):
    s=str(s).strip().casefold().replace("ö","o")
    return re.sub(r"[^0-9a-z가-힣]+","",s)

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

seed=load_seed()
visual=load_visual()
mapping=load_mapping()

st.markdown("### 고정된 확인표본")
st.dataframe(seed[["Brand","Seed_URL","URL_Status"]],use_container_width=True,hide_index=True)

st.markdown("### 고정 분석규칙")
st.code(
"""Text model: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
Hypothesis: This brand is {trait}.
Score: P(entailment)
Visual: frozen generic CLIP wordmark 42-trait
Primary similarity: Pearson r across raw 42-trait profiles
Primary H1 statistic: Mean Retrieval Rank
Permutation: 100,000 one-sided
alpha = .05
minimum confirmatory N = 15"""
)

uploaded=st.file_uploader(
    "최종 confirmation sentence/proposition corpus CSV 업로드",
    type=["csv"],
    help="필수열: Brand, Sentence_Unit_ID, Language, Sentence_Text"
)

if uploaded is not None:
    corpus=pd.read_csv(uploaded)
    required={"Brand","Sentence_Unit_ID","Language","Sentence_Text"}
    missing=required-set(corpus.columns)
    if missing:
        st.error(f"필수 열 누락: {sorted(missing)}")
        st.stop()

    allowed={bkey(x) for x in seed["Brand"]}
    corpus["_key"]=corpus["Brand"].map(bkey)
    extra=sorted(set(corpus["_key"])-allowed)
    if extra:
        st.error("확인표본 21개 이외 브랜드가 포함되어 있습니다. 개발표본과 섞지 마세요.")
        st.stop()

    counts=corpus.groupby("Brand").size().rename("N_Units").reset_index()
    st.dataframe(counts,use_container_width=True,hide_index=True)

    if st.button("H1 확인분석 실행",type="primary"):
        tok,model,device,labels=load_model()
        trait_keys=mapping["Trait_Key"].tolist()
        hypotheses=[f"This brand is {t}." for t in mapping["Trait"]]

        rows=[]
        total=len(corpus)*len(mapping)
        prog=st.progress(0.0,text=f"NLI {total:,} pairs")
        done=0
        # Process one trait at a time for deterministic memory use.
        for j,(_,a) in enumerate(mapping.iterrows()):
            premises=corpus["Sentence_Text"].astype(str).tolist()
            hyps=[hypotheses[j]]*len(premises)
            vals=nli_entailment(tok,model,device,labels,premises,hyps)
            for idx,val in enumerate(vals):
                rows.append({
                    "Brand":corpus.iloc[idx]["Brand"],
                    "Sentence_Unit_ID":corpus.iloc[idx]["Sentence_Unit_ID"],
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

        # Align to frozen visuals, preserving exact brand pairing.
        T=brand_trait.copy(); V=visual.copy()
        T["_key"]=T["Brand"].map(bkey); V["_key"]=V["Brand"].map(bkey)
        common=sorted(set(T["_key"])&set(V["_key"]))
        T=T.set_index("_key").loc[common].reset_index()
        V=V.set_index("_key").loc[common].reset_index()

        if len(common)<15:
            status="UNDER-SIZED CONFIRMATION"
            st.error(f"분석가능 브랜드 {len(common)}개 < 15개. Confirmatory 판정으로 사용하지 않습니다.")
        else:
            status="CONFIRMATORY"

        text_cols=trait_keys
        visual_cols=["Raw_"+x for x in trait_keys]
        A=T[text_cols].to_numpy(float)
        B=V[visual_cols].to_numpy(float)
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
            "N_Confirmed_Brands":n,
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
            "H1_Supported_Primary":bool(status=="CONFIRMATORY" and p_rank<0.05 and mean_rank<(n+1)/2),
        }])

        sim_df=pd.DataFrame(M,index=T["Brand"],columns=V["Brand"]).reset_index(names="Text_Brand")

        st.header("Confirmation 결과")
        st.dataframe(summary,use_container_width=True,hide_index=True)
        st.dataframe(detail.sort_values("Retrieval_Rank"),use_container_width=True,hide_index=True)

        files={
            "02C_01_unit_trait_scores.csv":unit_trait,
            "02C_02_brand_trait_profiles.csv":brand_trait,
            "02C_03_similarity_matrix.csv":sim_df,
            "02C_04_brand_retrieval_detail.csv":detail,
            "02C_05_confirmation_summary.csv":summary,
        }
        meta={
            "App_Version":APP_VERSION,
            "Model":MODEL_NAME,
            "Hypothesis":"This brand is {trait}.",
            "Text_Score":"P(entailment)",
            "Visual_Profile":"frozen generic CLIP wordmark raw 42-trait",
            "Primary_Similarity":"Pearson r across raw 42-trait profiles",
            "Primary_Statistic":"Mean Retrieval Rank",
            "Permutation_N":PRIMARY_PERMUTATIONS,
            "Permutation_Seed":RANDOM_SEED,
            "Alpha":0.05,
            "Minimum_Confirmatory_N":15,
            "Status":status,
        }
        st.download_button(
            "Confirmation 결과 ZIP 다운로드",
            data=zip_outputs(files,meta),
            file_name="02C_H1_confirmation_retrieval_results.zip",
            mime="application/zip"
        )
