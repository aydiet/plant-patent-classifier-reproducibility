from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import platform

import langid
import numpy as np
import tqdm
import tokenizers as _tk


# CoreML is a heavy import; defer to first use to avoid import-time hangs.
ct = None  # lazy-loaded


_SPLIT_PARA_RE = re.compile(r"(\n\s*\n+)")

# Sentence breaks: include . ! ? ; : plus common CJK punctuation.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.!\?;:\u3002\uff01\uff1f])\s+")


ISO_TO_NLLB: dict[str, str] = {
    # Common in your unsupported set
    "el": "ell_Grek",
    "hu": "hun_Latn",
    "no": "nob_Latn",  # Norwegian Bokmål
    "da": "dan_Latn",
    "sv": "swe_Latn",
    "fi": "fin_Latn",
    "ro": "ron_Latn",
    "cs": "ces_Latn",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "bg": "bul_Cyrl",
    "hr": "hrv_Latn",
    "sr": "srp_Latn",  # may override to srp_Cyrl depending on script
    "lv": "lvs_Latn",
    "lt": "lit_Latn",
    "et": "est_Latn",
    "is": "isl_Latn",
    "me": "cnr_Latn",  # Montenegrin
    "sh": "srp_Latn",  # Serbo-Croatian (approx)

    # A few common overall
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "tr": "tur_Latn",
    "pl": "pol_Latn",
    "nl": "nld_Latn",
    "ar": "arb_Arab",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "id": "ind_Latn",
}


_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


@dataclass(frozen=True)
class Candidate:
    docdb_family_id: int
    field: str
    source_appln_id: int
    source_lg: str
    target_lg: str
    source_text: str
    source_text_sha256: str


@dataclass(frozen=True)
class ResultRow:
    docdb_family_id: int
    field: str
    source_appln_id: int
    source_lg: str
    target_lg: str
    source_text_sha256: str
    ok: bool
    translated_text: str | None
    error: str | None
    engine: str
    os_version: str
    created_at_utc: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Translate candidate titles/abstracts to English using cstr/nllb-200-coreml-256 (CoreML). "
            "Designed as a fallback for Apple Translation failures; outputs a join-compatible results JSONL."
        )
    )
    p.add_argument(
        "candidates_jsonl",
        help="Input candidates JSONL (metadata/family_translation_candidates.jsonl).",
    )
    p.add_argument(
        "out_jsonl",
        help="Output results JSONL (join-compatible).",
    )
    p.add_argument(
        "--model-dir",
        default="models/nllb-200-coreml-256",
        help="Local model directory containing *.mlpackage and tokenizer/.",
    )
    p.add_argument(
        "--apple-results-jsonl",
        default="metadata/family_translation_results.jsonl",
        help=(
            "If provided, only translate candidates whose sha256 is missing in apple results or has ok=false. "
            "This lets you use NLLB as a fallback for Apple failures."
        ),
    )
    p.add_argument(
        "--translate-all",
        action="store_true",
        help="Translate all candidates regardless of Apple results.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing out_jsonl by skipping already-completed sha256 rows.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of candidates to translate (0 = no limit).",
    )
    p.add_argument(
        "--max-len",
        type=int,
        default=256,
        help="Max tokens for NLLB model (default: 256).",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=32,
        help="Token overlap when falling back to sliding windows for too-long sentences (default: 32).",
    )
    p.add_argument(
        "--compute-units",
        choices=["all", "cpu_only"],
        default="all",
        help="Core ML compute units (default: all).",
    )
    return p.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_candidates(path: Path) -> Iterable[Candidate]:
    for o in iter_jsonl(path):
        yield Candidate(
            docdb_family_id=int(o["docdb_family_id"]),
            field=str(o["field"]),
            source_appln_id=int(o["source_appln_id"]),
            source_lg=str(o.get("source_lg") or ""),
            target_lg=str(o.get("target_lg") or "en"),
            source_text=str(o.get("source_text") or ""),
            source_text_sha256=str(o.get("source_text_sha256") or ""),
        )


def load_needed_keys_from_apple_results(path: Path) -> set[tuple[int, str]]:
    needed: set[tuple[int, str]] = set()
    if not path.exists():
        return needed
    for o in iter_jsonl(path):
        fam = o.get("docdb_family_id")
        field = o.get("field")
        if fam is None or not isinstance(field, str):
            continue
        if not bool(o.get("ok")):
            needed.add((int(fam), field))
    return needed


def load_existing_done_keys(path: Path) -> set[tuple[int, str]]:
    done: set[tuple[int, str]] = set()
    if not path.exists():
        return done
    for o in iter_jsonl(path):
        fam = o.get("docdb_family_id")
        field = o.get("field")
        if fam is None or not isinstance(field, str):
            continue
        done.add((int(fam), field))
    return done


def normalize_whitespace(text: str) -> str:
    # Keep newlines for paragraph splitting; normalize inside paragraphs.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def split_paragraphs_preserve_delims(text: str) -> list[str]:
    # Split into [para, delim, para, delim, ...] so we can preserve blank-line boundaries.
    parts = _SPLIT_PARA_RE.split(text)
    # Keep as-is.
    return [p for p in parts if p != ""]


def split_sentences(paragraph: str) -> list[str]:
    # Normalize intra-paragraph whitespace to keep the segmenter stable.
    p = re.sub(r"\s+", " ", paragraph).strip()
    if not p:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(p) if s.strip()]


def guess_iso_lang(text: str) -> str:
    # langid returns ISO 639-1-ish codes.
    lg, _score = langid.classify(text)
    return lg


def iso_to_nllb(iso: str, text: str) -> str | None:
    iso = (iso or "").lower().strip()
    if not iso:
        return None

    if iso == "sr":
        # pick script-aware Serbian
        if _CYRILLIC_RE.search(text):
            return "srp_Cyrl"
        return "srp_Latn"

    return ISO_TO_NLLB.get(iso)


def ensure_sha256(text: str, existing: str) -> str:
    if existing:
        return existing
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class NLLBTranslator:
    def __init__(self, model_dir: Path, max_len: int, compute_units: str):
        global ct
        if ct is None:
            import coremltools as _ct
            ct = _ct

        self.model_dir = model_dir
        self.max_len = max_len

        units = ct.ComputeUnit.ALL if compute_units == "all" else ct.ComputeUnit.CPU_ONLY

        enc_path = model_dir / "NLLB_Encoder_256.mlpackage"
        dec_path = model_dir / "NLLB_Decoder_256.mlpackage"
        tok_path = model_dir / "tokenizer"

        if not enc_path.exists():
            raise FileNotFoundError(f"Missing encoder: {enc_path}")
        if not dec_path.exists():
            raise FileNotFoundError(f"Missing decoder: {dec_path}")
        if not tok_path.exists():
            raise FileNotFoundError(f"Missing tokenizer dir: {tok_path}")

        self.encoder = ct.models.MLModel(str(enc_path), compute_units=units)
        self.decoder = ct.models.MLModel(str(dec_path), compute_units=units)

        # Load tokenizer directly via the `tokenizers` library to avoid the
        # extremely slow `from transformers import AutoTokenizer` on this system.
        self._tok = _tk.Tokenizer.from_file(str(tok_path / "tokenizer.json"))

        # Build token→id look-up from the tokenizer vocabulary.
        vocab = self._tok.get_vocab()
        self._vocab = vocab
        self._pad_id: int = vocab.get("<pad>", 1)
        self._eos_id: int = vocab.get("</s>", 2)

    # ---- thin wrappers that replicate what AutoTokenizer did ----

    def _token_to_id(self, token: str) -> int:
        return self._vocab[token]

    def _encode_text(
        self,
        text: str,
        src_lang: str,
        *,
        add_special: bool = True,
        truncation: bool = False,
        pad_to: int | None = None,
    ) -> tuple[list[int], list[int]]:
        """Tokenise *text* and optionally add NLLB source-language framing.

        NLLB expected input:  <src_lang> tokens... </s>
        Returns (input_ids, attention_mask).
        """
        # Always encode without the tokenizer's built-in post-processor
        # (which hard-codes eng_Latn), then add correct src_lang manually.
        encoding = self._tok.encode(text, add_special_tokens=False)
        ids: list[int] = list(encoding.ids)

        if add_special:
            lang_id = self._vocab.get(src_lang)
            if lang_id is None:
                raise ValueError(f"Unknown NLLB language token: {src_lang}")
            ids = [lang_id] + ids + [self._eos_id]

        if truncation and len(ids) > self.max_len:
            ids = ids[: self.max_len]

        attn = [1] * len(ids)

        if pad_to is not None and len(ids) < pad_to:
            pad_n = pad_to - len(ids)
            ids = ids + [self._pad_id] * pad_n
            attn = attn + [0] * pad_n

        return ids, attn

    def _encode(self, text: str, src_lang: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids, attn = self._encode_text(
            text, src_lang, add_special=True, truncation=True, pad_to=self.max_len
        )
        input_ids = np.array([ids], dtype=np.int32)
        attention_mask = np.array([attn], dtype=np.int32)
        enc_out = self.encoder.predict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )
        encoder_hidden_states = enc_out[list(enc_out.keys())[0]]
        return (
            attention_mask,
            encoder_hidden_states,
            input_ids,
        )

    def _decode(self, encoder_hidden_states: np.ndarray, encoder_attention_mask: np.ndarray, target_lang: str) -> str:
        forced_bos = self._token_to_id(target_lang)
        # Follow the model repo's example.
        current_tokens: list[int] = [2, forced_bos]
        no_repeat_ngram_size = 3

        for _i in range(self.max_len - 2):
            decoder_input = np.full((1, self.max_len), self._pad_id, dtype=np.int32)
            decoder_input[0, : len(current_tokens)] = np.array(current_tokens, dtype=np.int32)

            dec_out = self.decoder.predict(
                {
                    "decoder_input_ids": decoder_input,
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_attention_mask": encoder_attention_mask,
                }
            )
            logits = dec_out[list(dec_out.keys())[0]]
            step_logits = logits[0, len(current_tokens) - 1, :]

            # Block tokens that would complete a repeated n-gram (size 3)
            if len(current_tokens) >= no_repeat_ngram_size:
                tail = tuple(current_tokens[-(no_repeat_ngram_size - 1):])
                for j in range(len(current_tokens) - no_repeat_ngram_size + 1):
                    window = tuple(current_tokens[j : j + no_repeat_ngram_size - 1])
                    if window == tail:
                        banned = current_tokens[j + no_repeat_ngram_size - 1]
                        step_logits[banned] = -1e9

            next_token = int(np.argmax(step_logits))
            if next_token == 2:
                break
            current_tokens.append(next_token)

        return self._tok.decode(current_tokens[2:], skip_special_tokens=False).strip()

    def translate_one(self, text: str, src_lang: str, tgt_lang: str) -> str:
        # If our chunking is correct, text should already be <= max_len.
        attn_mask, enc_hid, _ids = self._encode(text, src_lang)
        return self._decode(enc_hid, attn_mask, tgt_lang)

    def count_tokens(self, text: str, src_lang: str) -> int:
        ids, _ = self._encode_text(text, src_lang, add_special=True, truncation=False)
        return len(ids)

    def decode_token_window(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False).strip()

    def encode_no_trunc(self, text: str, src_lang: str) -> list[int]:
        ids, _ = self._encode_text(text, src_lang, add_special=False, truncation=False)
        return ids


def chunk_text(
    text: str,
    translator: NLLBTranslator,
    src_lang: str,
    max_len: int,
    overlap: int,
) -> list[str]:
    """Sentence-aware chunking with paragraph boundaries preserved.

    Returns a list of segment texts (each intended to fit within max_len tokens).
    Paragraph delimiters are preserved by inserting "\n\n" between paragraph translations.
    """

    norm = normalize_whitespace(text)
    parts = split_paragraphs_preserve_delims(norm)

    segments: list[str] = []

    for part in parts:
        if _SPLIT_PARA_RE.fullmatch(part):
            # paragraph delimiter
            segments.append("\n\n")
            continue

        sents = split_sentences(part)
        if not sents:
            continue

        cur: list[str] = []
        for sent in sents:
            candidate = (" ".join(cur + [sent])).strip() if cur else sent
            if translator.count_tokens(candidate, src_lang) <= max_len:
                cur = cur + [sent]
                continue

            # flush current
            if cur:
                segments.append(" ".join(cur).strip())
                cur = []

            # sentence itself might be too long: fallback to token windows
            if translator.count_tokens(sent, src_lang) <= max_len:
                cur = [sent]
            else:
                ids = translator.encode_no_trunc(sent, src_lang)
                if not ids:
                    continue
                step = max_len - overlap
                if step <= 0:
                    step = max_len
                for start in range(0, len(ids), step):
                    window = ids[start : start + max_len]
                    if not window:
                        continue
                    segments.append(translator.decode_token_window(window).strip())

        if cur:
            segments.append(" ".join(cur).strip())

    # Cleanup: collapse repeated delimiter segments
    cleaned: list[str] = []
    for s in segments:
        if s == "\n\n":
            if cleaned and cleaned[-1] == "\n\n":
                continue
        cleaned.append(s)

    # Ensure no leading/trailing delimiter
    while cleaned and cleaned[0] == "\n\n":
        cleaned = cleaned[1:]
    while cleaned and cleaned[-1] == "\n\n":
        cleaned = cleaned[:-1]

    return cleaned


def translate_candidate(
    cand: Candidate,
    translator: NLLBTranslator,
    max_len: int,
    overlap: int,
) -> tuple[bool, str | None, str | None, str]:
    """Returns (ok, translated_text, error, nllb_src_lang)."""

    text = cand.source_text or ""
    if not text.strip():
        return False, None, "missing_source_text", ""

    iso = cand.source_lg.strip().lower() if cand.source_lg else ""
    if not iso or iso in {"und", "unknown", "xx"}:
        iso = guess_iso_lang(text)

    nllb_src = iso_to_nllb(iso, text)
    if nllb_src is None:
        return False, None, "language_pair_unsupported", iso

    tgt = "eng_Latn"

    try:
        segments = chunk_text(text, translator, nllb_src, max_len=max_len, overlap=overlap)
        out_parts: list[str] = []
        for seg in segments:
            if seg == "\n\n":
                out_parts.append("\n\n")
                continue
            out_parts.append(translator.translate_one(seg, nllb_src, tgt))

        translated = "".join(out_parts)
        translated = re.sub(r"\n{3,}", "\n\n", translated).strip()
        return True, translated, None, iso
    except Exception as e:  # noqa: BLE001
        return False, None, f"translation_error: {type(e).__name__}: {e}", iso


def main() -> None:
    args = parse_args()

    candidates_path = Path(args.candidates_jsonl)
    out_path = Path(args.out_jsonl)
    model_dir = Path(args.model_dir)
    apple_results_path = Path(args.apple_results_jsonl) if args.apple_results_jsonl else None

    if not candidates_path.exists():
        raise SystemExit(f"Not found: {candidates_path}")

    needed_keys: set[tuple[int, str]] = set()
    if not args.translate_all and apple_results_path is not None:
        needed_keys = load_needed_keys_from_apple_results(apple_results_path)

    done_keys: set[tuple[int, str]] = set()
    if args.resume:
        done_keys = load_existing_done_keys(out_path)

    engine = "nllb200_coreml_256"
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    os_version = platform.platform()

    translator = NLLBTranslator(model_dir=model_dir, max_len=int(args.max_len), compute_units=args.compute_units)

    # Stream output JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    translated_count = 0
    skipped_count = 0
    total_seen = 0

    # Pre-filter candidates so progress reflects actual work.
    selected: list[Candidate] = []
    for cand in iter_candidates(candidates_path):
        key = (cand.docdb_family_id, cand.field)
        if args.resume and key in done_keys:
            continue
        if needed_keys and key not in needed_keys:
            continue
        selected.append(cand)

    pbar = tqdm.tqdm(selected, desc="NLLB translate", unit="cand")

    with out_path.open(mode, encoding="utf-8") as out_f:
        for cand in pbar:
            total_seen += 1
            sha = ensure_sha256(cand.source_text or "", cand.source_text_sha256)

            ok, translated_text, error, iso_used = translate_candidate(
                cand,
                translator,
                max_len=int(args.max_len),
                overlap=int(args.overlap),
            )

            row = ResultRow(
                docdb_family_id=cand.docdb_family_id,
                field=cand.field,
                source_appln_id=cand.source_appln_id,
                source_lg=iso_used,
                target_lg="en",
                source_text_sha256=sha,
                ok=ok,
                translated_text=translated_text,
                error=error,
                engine=engine,
                os_version=os_version,
                created_at_utc=created_at,
            )

            out_f.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")
            out_f.flush()

            translated_count += 1
            pbar.set_postfix(translated=translated_count)

            if args.limit and translated_count >= int(args.limit):
                break

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
