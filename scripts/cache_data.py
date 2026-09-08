import sys

from ssvep_b2b.cli import main

raise SystemExit(main(["cache", *sys.argv[1:]]))
