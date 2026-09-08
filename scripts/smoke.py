import sys

from ssvep_b2b.cli import main

raise SystemExit(main(["smoke", *sys.argv[1:]]))
