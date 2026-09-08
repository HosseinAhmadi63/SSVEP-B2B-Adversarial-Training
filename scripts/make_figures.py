import sys

from ssvep_b2b.cli import main

raise SystemExit(main(["figures", *sys.argv[1:]]))
