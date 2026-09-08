import sys

from ssvep_b2b.cli import main

raise SystemExit(main(["run-all", *sys.argv[1:]]))
