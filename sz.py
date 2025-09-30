#!/usr/bin/env python3
import os
import sys
import itertools
from tabulate import tabulate


def is_code_line(line: str) -> bool:
    line = line.strip()
    return bool(line) and not line.startswith('//') and not line.startswith('/*') and not line.startswith('*')


def count_tokens(line: str) -> int:
    return len(line.strip().split())


def gen_stats(base_path="."):
    table = []
    for subdir in ["core", "tests", "examples"]:
        for path, _, files in os.walk(os.path.join(base_path, subdir)):
            # for path, _, files in os.walk(os.path.join(base_path, "core")):
            for name in files:
                if not (name.endswith((".cpp", ".c", ".h", ".hpp"))):
                    continue

                # Normalize path to forward slashes
                norm_path = path.replace('\\', '/')

                # Exclude unwanted subdirs, but allow tests/STM32F411CEUx/Core
                if "Catch2" in norm_path or (
                    "tests/STM32F411CEUx" in norm_path
                    and "tests/STM32F411CEUx/Core" not in norm_path
                ):
                    continue

                filepath = os.path.join(path, name)
                relfilepath = os.path.relpath(
                    filepath, base_path).replace('\\', '/')
                with open(filepath, encoding="utf-8", errors="ignore") as file_:
                    lines = [line.strip() for line in file_.readlines()]
                    code_lines = [line for line in lines if is_code_line(line)]
                    token_count = sum(count_tokens(line)
                                      for line in code_lines)
                    line_count = len(code_lines)
                if line_count > 0:
                    table.append([relfilepath, line_count,
                                  token_count / line_count])
    return table


def gen_diff(table_old, table_new):
    table = []
    files_new = set([x[0] for x in table_new])
    files_old = set([x[0] for x in table_old])
    added, deleted, unchanged = files_new - \
        files_old, files_old - files_new, files_new & files_old
    if added:
        for file in added:
            file_stat = [stats for stats in table_new if file in stats]
            table.append([file_stat[0][0], file_stat[0][1], file_stat[0]
                         [1] - 0, file_stat[0][2], file_stat[0][2] - 0])
    if deleted:
        for file in deleted:
            file_stat = [stats for stats in table_old if file in stats]
            table.append([file_stat[0][0], 0, 0 - file_stat[0]
                         [1], 0, 0 - file_stat[0][2]])
    if unchanged:
        for file in unchanged:
            file_stat_old = [stats for stats in table_old if file in stats]
            file_stat_new = [stats for stats in table_new if file in stats]
            if file_stat_new[0][1] - file_stat_old[0][1] != 0 or file_stat_new[0][2] - file_stat_old[0][2] != 0:
                table.append([
                    file_stat_new[0][0],
                    file_stat_new[0][1],
                    file_stat_new[0][1] - file_stat_old[0][1],
                    file_stat_new[0][2],
                    file_stat_new[0][2] - file_stat_old[0][2]
                ])
    return table


def display_diff(diff): return f"+{diff}" if diff > 0 else str(diff)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        headers = ["Name", "Lines", "Diff", "Tokens/Line", "Diff"]
        table = gen_diff(gen_stats(sys.argv[1]), gen_stats(sys.argv[2]))
    elif len(sys.argv) == 2:
        headers = ["Name", "Lines", "Tokens/Line"]
        table = gen_stats(sys.argv[1])
    else:
        headers = ["Name", "Lines", "Tokens/Line"]
        table = gen_stats(".")

    if table:
        if len(sys.argv) == 3:
            print("### Changes")
            print("```")
            print(tabulate(
                [headers] + sorted(table, key=lambda x: -x[1]),
                headers="firstrow",
                intfmt=(..., "d", "+d"),
                floatfmt=(..., ..., ..., ".1f", "+.1f")
            ) + "\n")
            print(
                f"\ntotal lines changes: {display_diff(sum([x[2] for x in table]))}")
            print("```")
        else:
            print(tabulate([headers] + sorted(table, key=lambda x: -
                  x[1]), headers="firstrow", floatfmt=".1f") + "\n")
            groups = sorted(
                [('/'.join(x[0].rsplit("/", 1)[0].split("/")[0:2]), x[1], x[2]) for x in table])
            for dir_name, group in itertools.groupby(groups, key=lambda x: x[0]):
                print(f"{dir_name:30s} : {sum([x[1] for x in group]):6d}")
            total_lines = sum([x[1] for x in table])
            print(f"\ntotal line count: {total_lines}")
            max_line_count = int(os.getenv("MAX_LINE_COUNT", "-1"))
            assert max_line_count == - \
                1 or total_lines <= max_line_count, f"OVER {max_line_count} LINES"
