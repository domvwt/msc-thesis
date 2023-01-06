import argparse
from pathlib import Path


def get_parser():
    """Get the argument parser for this script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    return parser


def is_table_row(line):
    """Return True if line is a table row."""
    line = line.strip()
    return line.startswith("|") and line.endswith("|")


def fix_tables(text):
    """Fix markdown tables in text.

    If a line begins and ends with a pipe, it is a table row.
    If a table row is followed by a blank line and then another table
    row, we remove the blank line.
    """
    lines = text.splitlines()
    new_lines = []
    for i, line in enumerate(lines):
        if (
            is_table_row(lines[i - 1])
            and i + 1 < len(lines)
            and line.strip() == ""
            and is_table_row(lines[i + 1])
        ):
            continue
        new_lines.append(line)

    return "\n".join(new_lines)


def fix_bulleted_lists(text):
    """Fix markdown bulleted lists in text.

    If a line begins with a dash and is followed by a blank line and
    then another line beginning with a dash, we remove the blank line.
    """
    lines = text.splitlines()
    new_lines = []
    for i, line in enumerate(lines):
        if (
            lines[i - 1].strip().startswith("-")
            and i + 1 < len(lines)
            and line.strip() == ""
            and lines[i + 1].strip().startswith("-")
        ):
            continue
        new_lines.append(line)

    return "\n".join(new_lines)


def fix_yaml_front_matter(text):
    """Fix YAML front matter in text.

    There should be no blank lines between the opening and closing
    triple-dash lines.
    """
    lines = text.splitlines()
    new_lines = []
    front_matter_start = lines.index("---")
    front_matter_end = lines.index("---", front_matter_start + 1)
    for i, line in enumerate(lines):
        if i > front_matter_start and i < front_matter_end and line.strip() == "":
            continue
        new_lines.append(line)

    return "\n".join(new_lines)


def main():
    """Main entry point for the script."""
    parser = get_parser()
    args = parser.parse_args()
    text = args.file.read_text()
    text = fix_tables(text)
    text = fix_bulleted_lists(text)
    text = fix_yaml_front_matter(text)
    print(text)


if __name__ == "__main__":
    main()
