"""Back-compat forwarder. The CLI moved to ``motrack.cli.optimize``."""
from motrack.cli.optimize import main

if __name__ == '__main__':
    main()
