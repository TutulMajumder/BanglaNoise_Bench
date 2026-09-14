"""Phase 3 model registry -- the single source of truth for every model's
identity, size, and provenance. `pipeline/p3_train.py` reads this to build
and tokenize models; `pipeline/p3_figures.py` reads it to render the paper's
model comparison table (Figure_Plan_Evaluation.md, "Model comparison table
rendered from the registry fields"). Write facts here once; nothing else
should hardcode a vocab size or parameter count.

Five models (METHODOLOGY.md M4 / Topic4_FULL_PAPER_PLAN.md §6) -- a sixth,
MuRIL, was proposed for this registry and rejected 2026-09-15 without
checking M4 first: redundant with XLM-R/mBERT on the multilingual axis, and
~3-4 GPU-h not justified against the 30 Sep deadline. See PHASE3_STATUS.md.

``vocab_size``, ``hidden_size``, ``n_layers``, ``n_heads``: fetched directly
via ``transformers.AutoConfig.from_pretrained(hf_id)`` (2026-09-15), not
guessed. ``param_count``: fetched by actually loading each base encoder
(``AutoModel.from_pretrained(hf_id)``, no task head) and
``sum(p.numel() for p in model.parameters())`` (2026-09-15) -- the
classification head added per task (``hidden_size * num_labels +
num_labels``, negligible: ~2.3K params for 3-class, ~1.5K for binary) is not
included, matching the usual "backbone size" convention in a model table.

Standard library + no third-party imports here (same "no I/O, pure data"
spirit as ``noisebench/plotstyle.py``) -- `transformers`/`torch` are
`pipeline/`-only dependencies, imported by the scripts that actually load a
model, not by this registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    key: str                       # short id used throughout results.csv, filenames, etc.
    hf_id: str | None              # HuggingFace model id; None for the classical baseline
    display_name: str              # for figure legends / the paper's model table
    kind: str                      # "transformer" or "classical"
    tokenizer_type: str            # "wordpiece" / "sentencepiece-unigram" / "char_wb-ngram"
    vocab_size: int
    param_count: int               # backbone only, no classification head; 0 for classical
    hidden_size: int | None
    n_layers: int | None
    n_heads: int | None
    pretraining_corpus: str
    why_included: str
    notes: str = ""


MODELS: dict[str, ModelSpec] = {
    "banglabert": ModelSpec(
        key="banglabert",
        hf_id="csebuetnlp/banglabert",
        display_name="BanglaBERT",
        kind="transformer",
        tokenizer_type="sentencepiece-unigram",
        vocab_size=32000,
        param_count=110_026_752,
        hidden_size=768,
        n_layers=12,
        n_heads=12,
        # TODO(verify): exact corpus size figure. csebuetnlp describe an
        # ELECTRA-base discriminator pretrained on a large Bangla web-crawl
        # corpus the paper calls "Bangla2B+" (Bhattacharjee et al. 2022,
        # "BanglaBERT: Language Model Pretraining and Benchmarks for
        # Low-Resource Language Understanding Evaluation in Bangla") --
        # confirm the exact GB/token-count figure against the paper/model
        # card before it goes in the paper's model table.
        pretraining_corpus="Bangla-only web-crawl corpus ('Bangla2B+', "
                           "csebuetnlp) -- TODO(verify) exact size",
        why_included="The primary subject: the Bangla-specific model expected "
                     "to show the largest robustness gap between clean and "
                     "noisy performance, since its subword vocabulary is "
                     "fit to clean Bangla orthography.",
    ),
    "banglishbert": ModelSpec(
        key="banglishbert",
        hf_id="csebuetnlp/banglishbert",
        display_name="BanglishBERT",
        kind="transformer",
        tokenizer_type="sentencepiece-unigram",
        vocab_size=32000,
        param_count=110_026_752,
        hidden_size=768,
        n_layers=12,
        n_heads=12,
        # TODO(verify): same corpus-size caveat as banglabert; csebuetnlp
        # describe this as pretrained additionally on code-mixed
        # Bangla-English data for code-mixed-aware representations.
        pretraining_corpus="Bangla web-crawl corpus + code-mixed "
                           "Bangla-English data (csebuetnlp) -- "
                           "TODO(verify) exact composition/size",
        why_included="Code-mixed-aware competitor to BanglaBERT -- same "
                     "architecture and vocab size, different pretraining "
                     "mix, isolating the effect of code-mixing exposure "
                     "rather than architecture.",
    ),
    "xlmr": ModelSpec(
        key="xlmr",
        hf_id="xlm-roberta-base",
        display_name="XLM-R (base)",
        kind="transformer",
        tokenizer_type="sentencepiece-unigram",
        vocab_size=250002,
        param_count=278_043_648,
        hidden_size=768,
        n_layers=12,
        n_heads=12,
        pretraining_corpus="CommonCrawl, 100 languages, ~2.5TB filtered text "
                           "(Conneau et al. 2020)",
        why_included="Multilingual arm with a large shared subword "
                     "vocabulary (250K) -- tests whether broad multilingual "
                     "coverage buys robustness that a Bangla-only vocabulary "
                     "lacks, or whether it dilutes Bangla-specific capacity.",
    ),
    "mbert": ModelSpec(
        key="mbert",
        hf_id="bert-base-multilingual-cased",
        display_name="mBERT",
        kind="transformer",
        tokenizer_type="wordpiece",
        vocab_size=119547,
        param_count=177_853_440,
        hidden_size=768,
        n_layers=12,
        n_heads=12,
        pretraining_corpus="Wikipedia, 104 languages (Devlin et al. 2019 "
                           "multilingual release)",
        why_included="Older multilingual baseline (WordPiece, not "
                     "SentencePiece; Wikipedia, not CommonCrawl) -- a "
                     "second multilingual reference point on a different "
                     "tokenizer/corpus combination than XLM-R.",
    ),
    "char_ngram": ModelSpec(
        key="char_ngram",
        hf_id=None,
        display_name="Char n-gram + LinearSVC",
        kind="classical",
        tokenizer_type="char_wb-ngram",
        vocab_size=0,          # fit per-task by TfidfVectorizer; not fixed
        param_count=0,         # no backbone; LinearSVC coefficients scale with fitted vocab
        hidden_size=None,
        n_layers=None,
        n_heads=None,
        pretraining_corpus="none -- fit from scratch on each task's training split",
        why_included="The scientific control. No subword tokenizer at all: "
                     "TF-IDF over character n-grams (char_wb, 2-5) has no "
                     "vocabulary that can go out-of-distribution the way a "
                     "BPE/SentencePiece/WordPiece vocabulary can. It should "
                     "degrade far more gently under orthographic noise. If "
                     "it doesn't, that is itself the finding, not a null "
                     "result to discard.",
        notes="scikit-learn TfidfVectorizer(analyzer='char_wb', "
              "ngram_range=(2,5)) + LinearSVC(C=1.0, "
              "class_weight='balanced', min_df=2) -- METHODOLOGY M4. "
              "Runs on CPU; no GPU-hour cost.",
    ),
}

#: Canonical iteration order -- figures/tables list models in this order
#: everywhere, matching Figure_Plan_Evaluation.md's STYLE dict order.
MODEL_ORDER: tuple[str, ...] = ("banglabert", "char_ngram", "banglishbert", "xlmr", "mbert")

assert set(MODEL_ORDER) == set(MODELS), "MODEL_ORDER must list exactly the registered models"


def transformer_keys() -> tuple[str, ...]:
    return tuple(k for k in MODEL_ORDER if MODELS[k].kind == "transformer")


def all_keys() -> tuple[str, ...]:
    return MODEL_ORDER
