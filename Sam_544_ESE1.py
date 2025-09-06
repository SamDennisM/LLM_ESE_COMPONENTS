
import os
import json
from typing import List, Dict, Tuple

import streamlit as st

# Hugging Face / Transformers
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
)

# Metrics
from rouge_score import rouge_scorer
import sacrebleu

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Multilingual NLP: NER · Summarization · Translation",
    page_icon="🌍",
    layout="wide",
)

st.markdown(
    """
    # 🌍 Multilingual NLP Toolkit

    Implemented with 🤗 **Transformers**. Supports Indic and foreign languages.

    **Tasks**
    1. **Named Entity Recognition (NER)** — multilingual model (works for English + many languages; Indic coverage varies)
    2. **Text Summarization** — multilingual summarizer trained on XL-Sum
    3. **Machine Translation** — English → French (MarianMT)
    4. **Evaluations** — ROUGE for summarization, SacreBLEU for translation, optional precision/recall/F1 for NER

    _Tip:_ You can switch models from the sidebar if you have a preferred checkpoint.
    """
)

# -----------------------------
# Sidebar: Model Selection
# -----------------------------
with st.sidebar:
    st.header("⚙️ Models & Settings")

    st.caption("You can override with any Hugging Face model ID.")

    # NER models
    ner_default = "Davlan/xlm-roberta-base-ner-hrl"  
    ner_model_id = st.text_input("NER model id", value=ner_default, help="Try 'dslim/bert-base-NER' for English-only.")

    
    sum_default = "csebuetnlp/mT5_multilingual_XLSum"
    sum_model_id = st.text_input("Summarization model id", value=sum_default, help="Multilingual summarizer")

    # Translation model (English ➜ French)
    mt_default = "Helsinki-NLP/opus-mt-en-fr"
    mt_model_id = st.text_input("EN→FR Translation model id", value=mt_default, help="MarianMT (fast, high quality)")

    st.divider()

    st.subheader("Generation Settings")
    max_len = st.slider("Max output length (tokens)", 16, 1024, 256, 8)
    do_sample = st.checkbox("Use sampling (stochastic)", value=False)
    temperature = st.slider("Temperature (if sampling)", 0.1, 2.0, 1.0, 0.1)


@st.cache_resource(show_spinner=False)
def load_ner(model_id: str):
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForTokenClassification.from_pretrained(model_id)
        nlp = pipeline("token-classification", model=model, tokenizer=tok, aggregation_strategy="simple")
        return nlp, None
    except Exception as e:
        return None, str(e)

@st.cache_resource(show_spinner=False)
def load_summarizer(model_id: str):
    try:
        nlp = pipeline("summarization", model=model_id)
        return nlp, None
    except Exception as e:
        return None, str(e)

@st.cache_resource(show_spinner=False)
def load_translator(model_id: str):
    try:
        nlp = pipeline("translation_en_to_fr", model=model_id)
        return nlp, None
    except Exception as e:
        return None, str(e)


def pretty_ent(ent: Dict) -> str:
    start = ent.get('start')
    end = ent.get('end')
    word = ent.get('word')
    label = ent.get('entity_group') or ent.get('entity')
    score = ent.get('score', 0.0)
    return f"{word} [{label}] ({score:.2f})"



T1, T2, T3, T4 = st.tabs([
    "🧩 Named Entity Recognition",
    "📝 Summarization",
    "🔁 EN→FR Translation",
    "📊 Evaluation",
])


with T1:
    st.subheader("🧩 Named Entity Recognition (Multilingual)")
    st.caption("Paste text in English or another supported language. Model choice affects coverage.")

    sample_ner = (
        "भारत के प्रधानमंत्री नरेंद्र मोदी ने नई दिल्ली में G20 शिखर सम्मेलन का उद्घाटन किया।\n"
        "Apple Inc. announced new products in California."
    )

    text_ner = st.text_area("Input Text", value=sample_ner, height=180)

    colN1, colN2 = st.columns([1, 1])
    with colN1:
        run_ner = st.button("Run NER", type="primary")
    with colN2:
        show_table = st.checkbox("Show table output", value=True)

    if run_ner and text_ner.strip():
        nlp_ner, err = load_ner(ner_model_id)
        if err:
            st.error(f"Failed to load NER model: {err}")
        else:
            with st.spinner("Tagging entities..."):
                ents = nlp_ner(text_ner)
            # Display
            st.markdown("### Entities")
            if show_table:
                st.dataframe(ents, use_container_width=True)
            # Inline highlight
            try:
                from annotated_text import annotated_text
                chunks: List = []
                last_idx = 0
                for ent in ents:
                    s, e = ent['start'], ent['end']
                    if s > last_idx:
                        chunks.append(text_ner[last_idx:s])
                    label = ent.get('entity_group') or ent.get('entity')
                    chunks.append((text_ner[s:e], label, f"{ent.get('score',0):.2f}"))
                    last_idx = e
                if last_idx < len(text_ner):
                    chunks.append(text_ner[last_idx:])
                st.markdown("#### Inline Highlight")
                annotated_text(*chunks)
            except Exception:
                st.info("Install 'st-annotated-text' for inline highlights: pip install st-annotated-text")


with T2:
    st.subheader("📝 Multilingual Summarization (mT5 · XL-Sum)")
    st.caption("Works for many languages including Hindi, Bengali, Tamil, Telugu, Marathi, English, and more.")

    sample_sum = (
        "भारत में हाल के वर्षों में डिजिटल भुगतान में तेज़ी से वृद्धि हुई है।\n"
        "UPI जैसे प्लेटफ़ॉर्म ने छोटे व्यापारियों और उपभोक्ताओं दोनों के लिए लेनदेन को आसान बना दिया है।\n"
        "इसके परिणामस्वरूप कैशलेस अर्थव्यवस्था की ओर एक महत्वपूर्ण बदलाव देखा जा रहा है।"
    )

    text_sum = st.text_area("Long text (any supported language)", value=sample_sum, height=220)

    colS1, colS2, colS3 = st.columns([1, 1, 1])
    with colS1:
        min_len = st.slider("Min summary length", 5, 400, 30, 5)
    with colS2:
        max_len_sum = st.slider("Max summary length", 10, 1024, 120, 10)
    with colS3:
        num_beams = st.slider("Beam search: num_beams", 1, 10, 4, 1)

    if st.button("Summarize ✨", type="primary") and text_sum.strip():
        summarizer, err = load_summarizer(sum_model_id)
        if err:
            st.error(f"Failed to load summarizer: {err}")
        else:
            with st.spinner("Generating summary..."):
                try:
                    out = summarizer(
                        text_sum,
                        max_length=max_len_sum,
                        min_length=min_len,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        num_beams=None if do_sample else num_beams,
                        truncation=True,
                    )
                    summary = out[0]['summary_text'] if isinstance(out, list) else str(out)
                except Exception as e:
                    st.error(f"Summarization failed: {e}")
                    summary = ""
            if summary:
                st.markdown("### Summary")
                st.text_area("", value=summary, height=160)


with T3:
    st.subheader("🔁 English → French Translation (MarianMT)")
    st.caption("Enter English text; output will be in French.")

    sample_mt = (
        "India has seen rapid growth in digital payments in recent years. "
        "Platforms like UPI have made transactions easier for both small merchants and consumers."
    )

    text_mt = st.text_area("English text", value=sample_mt, height=180)

    if st.button("Translate ➜", type="primary") and text_mt.strip():
        translator, err = load_translator(mt_model_id)
        if err:
            st.error(f"Failed to load translation model: {err}")
        else:
            with st.spinner("Translating..."):
                try:
                    out = translator(text_mt, max_length=max_len)
                    fr_text = out[0]['translation_text'] if isinstance(out, list) else str(out)
                except Exception as e:
                    st.error(f"Translation failed: {e}")
                    fr_text = ""
            if fr_text:
                st.markdown("### French Output")
                st.text_area("", value=fr_text, height=160)


with T4:
    st.subheader("📊 Simple Evaluations")
    st.markdown(
        """
        **Supported**
        - **Summarization:** ROUGE-1/2/L (requires reference/ground-truth summary)
        - **Translation:** SacreBLEU (requires reference French translation)
        - **NER:** Precision/Recall/F1 (requires list of gold entities)

        _Note:_ For NER, provide gold entities as JSON list of objects with `"text"` and `"label"`.
        We'll match by exact string on the system predictions.
        """
    )

    eval_task = st.selectbox("Task to evaluate", ["Summarization", "Translation", "NER"], index=0)

    if eval_task == "Summarization":
        st.markdown("#### ROUGE for Summarization")
        sys_sum = st.text_area("System summary")
        ref_sum = st.text_area("Reference summary (gold)")
        if st.button("Compute ROUGE"):
            if not sys_sum or not ref_sum:
                st.warning("Please provide both system and reference summaries.")
            else:
                scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
                scores = scorer.score(ref_sum, sys_sum)
                st.json({k: {"p": v.precision, "r": v.recall, "f": v.fmeasure} for k, v in scores.items()})

    elif eval_task == "Translation":
        st.markdown("#### SacreBLEU for EN→FR Translation")
        sys_mt = st.text_area("System translation (French)")
        ref_mt = st.text_area("Reference translation (French)")
        if st.button("Compute SacreBLEU"):
            if not sys_mt or not ref_mt:
                st.warning("Please provide both system and reference translations.")
            else:
                bleu = sacrebleu.corpus_bleu([sys_mt], [[ref_mt]])
                st.json({"sacrebleu": bleu.score})

    else:  # NER
        st.markdown("#### NER: Precision / Recall / F1")
        st.write("Paste system predictions and gold entities as JSON arrays.")
        st.caption("System predictions: from the NER tab output (list of dict, each with 'word' and 'entity_group')")

        sys_json = st.text_area("System entities (JSON)")
        gold_json = st.text_area("Gold entities (JSON) — list of {\"text\":..., \"label\":...}")
        label_sensitive = st.checkbox("Match requires same label", value=True)

        def to_set_system(preds: List[Dict]) -> set:
            s = set()
            for p in preds:
                txt = p.get('word') or p.get('text') or ""
                lab = p.get('entity_group') or p.get('entity') or ""
                key = (txt.strip(), lab.strip().upper())
                s.add(key)
            return s

        def to_set_gold(gold: List[Dict]) -> set:
            s = set()
            for g in gold:
                txt = g.get('text') or g.get('word') or ""
                lab = (g.get('label') or g.get('entity') or "").upper()
                key = (txt.strip(), lab)
                s.add(key)
            return s

        if st.button("Compute NER Scores"):
            try:
                sys_list = json.loads(sys_json)
                gold_list = json.loads(gold_json)
                sys_set = to_set_system(sys_list)
                gold_set = to_set_gold(gold_list)

                if not label_sensitive:
                    # Drop labels for matching
                    sys_set = {(t, "") for (t, l) in sys_set}
                    gold_set = {(t, "") for (t, l) in gold_set}

                tp = len(sys_set & gold_set)
                fp = len(sys_set - gold_set)
                fn = len(gold_set - sys_set)
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

                st.json({
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                })
            except Exception as e:
                st.error(f"Failed to compute NER metrics: {e}")


st.divider()
with st.expander("ℹ️ Notes & Tips", expanded=False):
    st.markdown(
        """
        - **Models**: Defaults are chosen for broad coverage and good quality. Swap in any Hugging Face model ID you prefer.
        - **Indic NER coverage** varies across checkpoints; try different multilingual NER models if results look sparse.
        - For **Summarization**, the XL-Sum fine-tuning generally works well for many languages.
        - For **Translation**, MarianMT en→fr is strong and fast. You can also try mBART-50.
        - If you run into CUDA OOM, reduce `max length`, `beam size`, or run on CPU.

        **Install**
        ```bash
        pip install streamlit transformers sentencepiece sacrebleu rouge-score st-annotated-text
        ```

        **Run**
        ```bash
        streamlit run streamlit_app.py
        ```
        """
    )
