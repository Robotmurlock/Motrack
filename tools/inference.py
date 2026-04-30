"""Back-compat forwarder. The CLI moved to ``motrack.cli.inference``."""
from motrack.cli.inference import main

if __name__ == '__main__':
    main()
