"""Run a frozen search controller; real calls still require --allow-api."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    if "--mode" in sys.argv and sys.argv[sys.argv.index("--mode") + 1] != "search":
        raise SystemExit("This entry point only supports frozen search")
    sys.argv += ["--mode", "search"]
    from graphskillevo.run_rl import main
    main()
