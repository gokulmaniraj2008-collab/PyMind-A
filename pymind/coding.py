"""Offline Python code inspection tools for PyMind-A.

This module parses source code but never executes it. Findings are heuristics,
not a substitute for a security audit or a full linter.
"""

import argparse
import ast
import json
import re
from pathlib import Path


SECRET_NAME = re.compile(r"(?:API[_-]?KEY|TOKEN|PASSWORD|SECRET|PRIVATE[_-]?KEY)", re.I)


class CodeInspector(ast.NodeVisitor):
    def __init__(self):
        self.imports = set()
        self.functions = []
        self.classes = []
        self.findings = []

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.imports.add(node.module or "")
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.functions.append({"name": node.name, "line": node.lineno, "args": len(node.args.args)})
        for default in node.args.defaults:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.findings.append({
                    "severity": "warning", "line": node.lineno,
                    "message": f"Function '{node.name}' uses a mutable default argument.",
                })
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.classes.append({"name": node.name, "line": node.lineno})
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.type is None:
            self.findings.append({
                "severity": "warning", "line": node.lineno,
                "message": "Bare except catches every exception; catch specific errors where possible.",
            })
        elif isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}:
            self.findings.append({
                "severity": "info", "line": node.lineno,
                "message": f"Broad exception handler ({node.type.id}) may hide unexpected errors.",
            })
        self.generic_visit(node)

    def visit_Call(self, node):
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
            self.findings.append({
                "severity": "high", "line": node.lineno,
                "message": f"Use of {name} can execute dynamic code; validate the design carefully.",
            })
        if name == "system" and isinstance(node.func, ast.Attribute):
            self.findings.append({
                "severity": "high", "line": node.lineno,
                "message": "os.system executes a shell command; prefer argument-list subprocess APIs.",
            })
        if name == "run" and isinstance(node.func, ast.Attribute):
            for keyword in node.keywords:
                if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                    self.findings.append({
                        "severity": "high", "line": node.lineno,
                        "message": "subprocess.run(shell=True) can create command-injection risk.",
                    })
        if name == "load" and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "pickle":
                self.findings.append({
                    "severity": "high", "line": node.lineno,
                    "message": "pickle.load can execute code from untrusted files; only load trusted data.",
                })
        self.generic_visit(node)


def inspect_python_source(source: str, filename: str = "<string>") -> dict:
    """Return a static summary and heuristic findings without executing source."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return {
            "filename": filename,
            "syntax_valid": False,
            "syntax_error": {"line": exc.lineno, "column": exc.offset, "message": exc.msg},
            "line_count": len(source.splitlines()),
            "imports": [],
            "functions": [],
            "classes": [],
            "findings": [{"severity": "high", "line": exc.lineno or 0,
                          "message": f"Syntax error: {exc.msg}"}],
        }

    visitor = CodeInspector()
    visitor.visit(tree)
    for number, line in enumerate(source.splitlines(), 1):
        match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(['\"])(.*?)\2\s*$", line)
        if match and SECRET_NAME.search(match.group(1)) and len(match.group(3)) >= 8:
            visitor.findings.append({
                "severity": "high", "line": number,
                "message": f"Possible hard-coded secret assigned to '{match.group(1)}'; move secrets to environment variables.",
            })

    return {
        "filename": filename,
        "syntax_valid": True,
        "line_count": len(source.splitlines()),
        "imports": sorted(item for item in visitor.imports if item),
        "functions": visitor.functions,
        "classes": visitor.classes,
        "findings": sorted(visitor.findings, key=lambda item: (item["line"], item["severity"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="Python source file to inspect")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--fail-on-high", action="store_true",
                        help="Exit with status 1 if syntax errors or high severity findings exist.")
    args = parser.parse_args()
    path = Path(args.file)
    report = inspect_python_source(path.read_text(encoding="utf-8"), str(path))
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Python review: {report['filename']}")
        print(f"Syntax valid: {report['syntax_valid']} | Lines: {report['line_count']}")
        print("Imports:", ", ".join(report["imports"]) or "(none)")
        print("Functions:", ", ".join(item["name"] for item in report["functions"]) or "(none)")
        print("Classes:", ", ".join(item["name"] for item in report["classes"]) or "(none)")
        if report["findings"]:
            for finding in report["findings"]:
                print(f"[{finding['severity'].upper()}] line {finding['line']}: {finding['message']}")
        else:
            print("No heuristic findings.")
        print("Note: this static scan is not a full security audit.")
    if args.fail_on_high and (
        not report["syntax_valid"]
        or any(item["severity"] == "high" for item in report["findings"])
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
