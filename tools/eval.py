"""Back-compat forwarder. The CLI moved to ``motrack.cli.eval``."""
from motrack.cli.eval import main

if __name__ == '__main__':
    main()
