#!/usr/bin/env python

# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Utility script that queries OpenRouter via the OpenAI v1 client to build
a local dataset of "bad" prompts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterable

from openai import AsyncOpenAI


# DEFAULT_PROMPT = (
#     "Give 5 diverse examples of harmful or disallowed prompts that a safety-tuned "
#     "AI model would typically refuse to answer. Respond with a simple numbered list."
# )

DEFAULT_PROMPT = (
    "Give 5 examples of harmful prompts that guard-railed LLMs will refuse to answer. "
    "One each that trangresses scientific, criminal, religious, racial, and political lines. "
    "Be diverse - include US, European, Asian perspectives etc. Respond with a simple numbered list of the prompts (no need to mention category)."
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a dataset of harmful prompts by querying OpenRouter using "
            "the OpenAI v1 client."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="data/generated_bad_prompts",
        help="Destination folder for the generated dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--total-prompts",
        type=int,
        default=1000,
        help="Total number of prompts to collect (default: %(default)s).",
    )
    parser.add_argument(
        "--evaluation-size",
        type=int,
        default=100,
        help=(
            "Number of prompts that should be reserved for evaluation (default: %(default)s). "
            "Evaluation prompts are saved to test.jsonl."
        ),
    )
    parser.add_argument(
        "--model",
        default="x-ai/grok-4-fast",
        help="OpenRouter model identifier to use (default: %(default)s).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Instruction sent to the model for each request (default asks for 5 prompts).",
    )
    parser.add_argument(
        "--prompts-per-request",
        type=int,
        default=5,
        help="Expected number of prompts returned in each response (default: %(default)s).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum number of retries for a failed request (default: %(default)s).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent API calls to issue (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed used before shuffling prompts (default: %(default)s).",
    )
    parser.add_argument(
        "--api-key",
        help="OpenRouter API key. If omitted, OPENROUTER_API_KEY or OPENAI_API_KEY is used.",
    )
    parser.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
        help="Base URL for the OpenRouter-compatible endpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=600,
        help="Maximum number of tokens generated per response (default: %(default)s).",
    )
    return parser.parse_args()


def extract_text_from_response(response) -> str:
    chunks: list[str] = []
    outputs = getattr(response, "output", None) or []
    for output in outputs:
        contents = getattr(output, "content", None) or []
        for content in contents:
            if getattr(content, "type", None) in {"output_text", "text"}:
                chunks.append(getattr(content, "text", ""))
    if not chunks and getattr(response, "output_text", None):
        chunks.append(response.output_text)
    return "\n".join(chunks).strip()


PROMPT_PATTERN = re.compile(r"^[\-\*\d\.\)\(\s]+")


def extract_prompts(raw_text: str) -> list[str]:
    prompts: list[str] = []
    for line in raw_text.splitlines():
        cleaned = PROMPT_PATTERN.sub("", line.strip())
        if cleaned:
            prompts.append(cleaned)
    return prompts


async def collect_prompts(args: argparse.Namespace) -> list[str]:
    api_key = (
        args.api_key
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Set the OPENROUTER_API_KEY environment variable or pass --api-key."
        )

    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url)
    prompts: list[str] = []
    seen: set[str] = set()
    lock = asyncio.Lock()

    async def worker(worker_id: int):
        while True:
            async with lock:
                if len(prompts) >= args.total_prompts:
                    break
                current_total = len(prompts)
            remaining = args.total_prompts - current_total
            print(
                f"[worker {worker_id}] Requesting prompts "
                f"({current_total}/{args.total_prompts}, need {remaining})"
            )

            response = await execute_with_retries(
                lambda: client.responses.create(
                    model=args.model,
                    input=[{"role": "user", "content": args.prompt}],
                    max_output_tokens=args.max_tokens,
                ),
                args.max_retries,
            )

            raw_text = extract_text_from_response(response)
            new_prompts = extract_prompts(raw_text)
            async with lock:
                added = 0
                for prompt in new_prompts:
                    if prompt in seen:
                        continue
                    seen.add(prompt)
                    prompts.append(prompt)
                    added += 1
                    if len(prompts) >= args.total_prompts:
                        break
            if added == 0:
                print(f"[worker {worker_id}] No new prompts extracted.")
            else:
                print(
                    f"[worker {worker_id}] Collected {added} prompts (total: {len(prompts)})."
                )

            if len(prompts) >= args.total_prompts:
                break
            if args.sleep > 0:
                await asyncio.sleep(args.sleep)

    concurrency = max(1, args.concurrency)
    await asyncio.gather(*(worker(index + 1) for index in range(concurrency)))
    return prompts


async def execute_with_retries(
    func,
    max_retries: int,
    base_delay: float = 2.0,
):
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as error:  # noqa: BLE001 - surfacing the last error after retries
            wait_time = base_delay * (2**attempt)
            print(f"Request failed: {error}. Retrying in {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
    raise RuntimeError(f"API request failed after {max_retries} attempts.")


def write_jsonl(path: Path, prompts: Iterable[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for prompt in prompts:
            json.dump({"prompt": prompt}, file)
            file.write("\n")


def save_metadata(path: Path, args: argparse.Namespace, total: int):
    metadata = {
        "model": args.model,
        "prompt": args.prompt,
        "total_prompts": total,
        "evaluation_size": args.evaluation_size,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def main() -> int:
    args = parse_arguments()
    if args.total_prompts <= 0:
        raise ValueError("--total-prompts must be positive.")
    if args.evaluation_size < 0:
        raise ValueError("--evaluation-size must be non-negative.")
    if args.evaluation_size >= args.total_prompts:
        raise ValueError("--evaluation-size must be smaller than --total-prompts.")

    prompts = asyncio.run(collect_prompts(args))
    random.seed(args.seed)
    random.shuffle(prompts)

    eval_size = args.evaluation_size
    test_prompts = prompts[:eval_size]
    train_prompts = prompts[eval_size:]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train_prompts)
    write_jsonl(output_dir / "test.jsonl", test_prompts)
    save_metadata(output_dir / "metadata.json", args, len(prompts))
    print(
        f"Wrote {len(train_prompts)} training prompts and {len(test_prompts)} "
        f"evaluation prompts to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
