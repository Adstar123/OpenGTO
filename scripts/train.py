"""Training script using the new CLI."""
import sys
from pathlib import Path

# Add parent directory to path so we can import poker_gto
sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_gto.cli import main

if __name__ == '__main__':
    # Replace script name and insert 'train' command
    sys.argv[0] = 'opengto'
    if len(sys.argv) == 1 or not sys.argv[1].startswith('-'):
        sys.argv.insert(1, 'train')
    else:
        # If there are arguments but no command, insert 'train' at position 1
        sys.argv.insert(1, 'train')
    main()