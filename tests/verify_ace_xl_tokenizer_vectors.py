"""Offline CPU check; run with Transformers and the pinned tokenizer directory.

Usage: python tests/verify_ace_xl_tokenizer_vectors.py /path/to/ace/tokenizer
No model weights, inference, global settings, or network access are involved.
"""

import ast
import hashlib
import json
import os
import sys
import types
from pathlib import Path

from transformers import AutoTokenizer


def verify(directory):
    plugin = Path(__file__).resolve().parents[1]
    evidence = json.loads((plugin / "tests/fixtures/music_song_v1_tokenizer_evidence_draft.json").read_text())
    directory = Path(directory)
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == evidence["asset_sha256"]["ace/" + filename]
    root = Path(os.environ.get("WAN2GP_ROOT") or plugin.parents[1])
    source = root / "models/TTS/ace_step15/pipeline_ace_step15.py"
    tree, namespace = ast.parse(source.read_text(encoding="utf-8-sig")), {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in ("_SFT_GEN_PROMPT", "_DEFAULT_DIT_INSTRUCTION") for target in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    pipeline = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ACEStep15Pipeline")
    helpers = types.SimpleNamespace()
    for node in pipeline.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("_format_meta", "_build_text_prompt", "_build_lyrics_prompt"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
            setattr(helpers, node.name, types.MethodType(namespace[node.name], helpers))
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    controls = evidence["accounting"]["xl_controls"]
    checked = []
    for vector in evidence["vectors"]:
        if not vector["case"].startswith("xl_"):
            continue
        if "description" in vector["case"]:
            meta = helpers._format_meta(controls["bpm"], controls["duration_seconds"], controls["key"], controls["time_signature"])
            prompt = helpers._build_text_prompt(vector["input"], meta)
        else:
            prompt = helpers._build_lyrics_prompt(vector["input"], controls["language"])
        assert prompt == vector["assembled_prompt"], vector["case"]
        ids = tokenizer(prompt, padding=False, truncation=False)["input_ids"]
        assert ids == vector["input_ids"] and len(ids) == vector["assembled_tokens"], vector["case"]
        assert (len(ids) <= vector["budget"]) is ("overflow" not in vector["case"])
        checked.append({"case": vector["case"], "tokens": len(ids), "budget": vector["budget"]})
    assert len(checked) == 4
    print(json.dumps({"status": "passed_cpu_native_vectors_not_gpu_qualification", "checks": checked}))


if __name__ == "__main__":
    verify(sys.argv[1])
