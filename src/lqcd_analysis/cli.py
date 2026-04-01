from pathlib import Path


def main() -> None:
    root = Path.cwd()
    print(f"lqcd-analysis scaffold ready at {root}")


if __name__ == "__main__":
    main()

