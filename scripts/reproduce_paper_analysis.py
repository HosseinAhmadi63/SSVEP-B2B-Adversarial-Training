import sys

from ssvep_b2b.cli import main

raise SystemExit(main(["reproduce-paper-analysis", *sys.argv[1:]]))
