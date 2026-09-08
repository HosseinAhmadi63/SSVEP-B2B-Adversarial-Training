import sys

from ssvep_b2b.cli import main

raise SystemExit(main(["download", *sys.argv[1:]]))
