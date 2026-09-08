import sys

from ssvep_b2b.cli import main

raise SystemExit(main(["run", *sys.argv[1:]]))
