#!/usr/bin/env python

# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image, UnidentifiedImageError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from unsloth import FastVisionModel
from transformers import AutoProcessor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with a Qwen3-VL LoRA checkpoint using optional images."
    )
    parser.add_argument(
        "--adapters",
        type=Path,
        help="Path to a saved LoRA snapshot. Defaults to the newest directory in --adapters-dir.",
    )
    parser.add_argument(
        "--adapters-dir",
        type=Path,
        default=Path("outputs/saved_models"),
        help="Directory containing saved LoRA snapshots (default: %(default)s).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/images"),
        help="Folder scanned for selectable images (default: %(default)s).",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="System message injected at the start of the conversation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to decode per reply (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: %(default)s).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling (default: %(default)s).",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Disable sampling and use greedy decoding.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="device_map passed to FastVisionModel.from_pretrained (default: %(default)s).",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Force bfloat16 weights when available.",
    )
    return parser.parse_args()


def discover_latest_snapshot(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = [item for item in root.iterdir() if item.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime)
    return candidates[-1]


def validate_adapter_path(args: argparse.Namespace, console: Console) -> Path:
    if args.adapters is not None:
        adapter_path = args.adapters.expanduser().resolve()
        if not adapter_path.exists():
            console.print(f"[red]Adapter directory {adapter_path} does not exist.[/]")
            sys.exit(1)
        return adapter_path
    root = args.adapters_dir.expanduser().resolve()
    latest = discover_latest_snapshot(root)
    if latest is None:
        console.print(
            f"[red]No LoRA snapshots found. Specify --adapters or create one under {root}.[/]"
        )
        sys.exit(1)
    return latest


def human_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    unit = units[0]
    for candidate in units[1:]:
        if value < 1024:
            break
        value /= 1024
        unit = candidate
    return f"{value:.1f} {unit}"


def list_image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    files = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(path)
    return files


def choose_image(console: Console, image_paths: list[Path]) -> Path | None:
    if not image_paths:
        console.print("[yellow]No images available in the configured folder.[/]")
        return None
    table = Table(title="Available images", show_lines=False)
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Filename", style="white")
    table.add_column("Size", justify="right", style="magenta")
    for index, path in enumerate(image_paths, start=1):
        try:
            size = human_size(path.stat().st_size)
        except OSError:
            size = "?"
        table.add_row(str(index), path.name, size)
    console.print(table)
    while True:
        selection = Prompt.ask(
            "Select an image number, or enter N for no image", default="N"
        ).strip()
        lowered = selection.lower()
        if lowered in {"n", "none", ""}:
            return None
        if lowered in {"r", "refresh"}:
            return None
        if selection.isdigit():
            choice = int(selection) - 1
            if 0 <= choice < len(image_paths):
                return image_paths[choice]
        console.print("[red]Invalid selection. Try again.[/]")


def build_template_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    templated: list[dict[str, object]] = []
    for message in messages:
        content = []
        for block in message.get("content", []):
            if block.get("type") == "image":
                content.append({"type": "image"})
            else:
                content.append({"type": "text", "text": block.get("text", "")})
        templated.append({"role": message["role"], "content": content})
    return templated


def collect_images(messages: Iterable[dict[str, object]]) -> list[Image.Image]:
    payloads: list[Image.Image] = []
    for message in messages:
        for block in message.get("content", []):
            if block.get("type") == "image" and block.get("image") is not None:
                payloads.append(block["image"])
    return payloads


def attach_image_to_message(path: Path | None, console: Console) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        with Image.open(path) as original:
            image = original.convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as error:
        console.print(f"[red]Failed to load {path}: {error}[/]")
        return None
    image_block = {"type": "image", "image": image, "path": str(path)}
    return image_block


def generate_response(
    model,
    processor,
    messages: list[dict[str, object]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> str:
    templated = build_template_messages(messages)
    chat_text = processor.apply_chat_template(
        templated,
        add_generation_prompt=True,
        tokenize=False,
    )
    processor_kwargs = {
        "text": [chat_text],
        "return_tensors": "pt",
        "padding": True,
    }
    images = collect_images(messages)
    if images:
        processor_kwargs["images"] = images
    inputs = processor(**processor_kwargs)
    device = next(model.parameters()).device
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    prompt_length = inputs["input_ids"].shape[-1]
    response_ids = output[:, prompt_length:]
    text = processor.tokenizer.decode(
        response_ids[0],
        skip_special_tokens=True,
    )
    return text.strip()


def main() -> None:
    args = parse_arguments()
    console = Console()
    adapter_path = validate_adapter_path(args, console)
    console.print(f"[cyan]Loading adapters from {adapter_path}[/cyan]")
    dtype = torch.bfloat16 if args.bf16 else None
    model, tokenizer = FastVisionModel.from_pretrained(
        str(adapter_path),
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
        device_map=args.device_map,
        dtype=dtype,
    )
    tokenizer.padding_side = "left"
    FastVisionModel.for_inference(model)
    try:
        processor = AutoProcessor.from_pretrained(adapter_path)
    except Exception as error:
        console.print(f"[red]Failed to load processor from {adapter_path}: {error}[/]")
        sys.exit(1)
    console.print("[green]Model ready.[/green]")
    image_paths = list_image_files(args.images_dir.expanduser())
    if image_paths:
        console.print(
            f"[cyan]Found {len(image_paths)} image(s) inside {args.images_dir}[/cyan]"
        )
    else:
        console.print(
            f"[yellow]No images detected in {args.images_dir}. Use /image after adding files.[/yellow]"
        )
    current_image: Path | None = None
    console.print(
        "[dim]Commands: /image choose file, /refresh rescan folder, "
        "/none clear image, /exit quit.[/dim]"
    )
    while True:
        if current_image is None:
            console.print("[dim]Current image: none[/dim]")
        else:
            console.print(f"[dim]Current image: {current_image}[/dim]")
        user_input = Prompt.ask("You").strip()
        if not user_input:
            console.print("[yellow]Empty input detected, exiting.[/]")
            break
        lowered = user_input.lower()
        if lowered in {"/exit", "/quit", "/q"}:
            console.print("[yellow]Goodbye.[/]")
            break
        if lowered in {"/image", "/img"}:
            current_image = choose_image(console, image_paths)
            continue
        if lowered == "/refresh":
            image_paths = list_image_files(args.images_dir.expanduser())
            console.print(f"[cyan]Rescanned. {len(image_paths)} image(s) available.[/cyan]")
            continue
        if lowered == "/none":
            current_image = None
            continue
        image_block = attach_image_to_message(current_image, console)
        messages: list[dict[str, object]] = []
        if args.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": args.system_prompt}],
                }
            )
        user_content = [{"type": "text", "text": user_input}]
        if image_block is not None:
            user_content.insert(0, image_block)
        messages.append({"role": "user", "content": user_content})
        try:
            response_text = generate_response(
                model=model,
                processor=processor,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=not args.greedy,
            )
        except KeyboardInterrupt:
            console.print("[red]Generation interrupted.[/]")
            break
        except Exception as error:
            console.print(f"[red]Generation failed: {error}[/]")
            continue
        console.print(
            Panel.fit(
                response_text or "[dim]No output[/dim]",
                title="Assistant",
                border_style="green",
            )
        )


if __name__ == "__main__":
    main()
