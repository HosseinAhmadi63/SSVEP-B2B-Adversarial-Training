import sys

from ssvep_b2b.cli import main

raise SystemExit(main(["verify-paper", *sys.argv[1:]]))
