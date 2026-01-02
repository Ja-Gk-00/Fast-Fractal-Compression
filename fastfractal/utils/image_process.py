import argparse
from pathlib import Path

from PIL import Image


def process_image(
    in_path: Path,
    out_path: Path,
    size: tuple[int, int] | None = None,
    grayscale: bool = False,
    to_format: str | None = None,
    jpg_quality: int = 95,
) -> None:
    img = Image.open(in_path)
    if grayscale:
        img = img.convert("L")  # type: ignore[assignment]
    else:
        img = img.convert("RGB")  # type: ignore[assignment]

    if size is not None:
        img = img.resize(size)  # type: ignore[assignment]

    if to_format is not None:
        fmt = to_format.upper()
        out_path = out_path.with_suffix(f".{fmt.lower()}")
    else:
        fmt = img.format or out_path.suffix.replace(".", "").upper()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {}
    if fmt == "JPG" or fmt == "JPEG":
        save_kwargs["quality"] = jpg_quality
        save_kwargs["subsampling"] = 0
        save_kwargs["optimize"] = True

    img.save(out_path, fmt, **save_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Input image or directory")
    parser.add_argument("output", type=Path, help="Output directory")

    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--to-format", choices=["png", "jpg", "jpeg"])
    parser.add_argument("--jpg-quality", type=int, default=95)

    args = parser.parse_args()

    size = None
    if args.width and args.height:
        size = (args.width, args.height)

    paths = []
    if args.input.is_dir():
        paths = [
            p
            for p in args.input.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    else:
        paths = [args.input]

    for p in paths:
        rel = p.relative_to(args.input) if args.input.is_dir() else p.name
        out = args.output / rel
        process_image(
            p,
            out,
            size=size,
            grayscale=args.grayscale,
            to_format=args.to_format,
            jpg_quality=args.jpg_quality,
        )


if __name__ == "__main__":
    main()
