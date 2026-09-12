"""Check the Git index and reachable history for private files before publishing.

Prints paths/revisions only, never matching secret contents. Exit 1 blocks release.
"""
import pathlib
import re
import subprocess


def git(*args):
    return subprocess.check_output(["git", *args])


def forbidden(path):
    p = pathlib.PurePosixPath(path.lower())
    return (
        p.parts[0] == "datasets"
        or any(x in {"logs", "outputs", ".codex", ".vscode", "pet"} for x in p.parts)
        or p.name.startswith(("pet_", ".env")) or p.name == "pet.py"
        or p.suffix in {".pt", ".pth", ".ckpt", ".onnx", ".bag", ".db3", ".mcap", ".pem", ".key"}
    )


def main():
    failures = set()
    index = git("ls-files", "-z").decode().split("\0")
    patterns = [
        rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
        rb"gh[pousr]_[A-Za-z0-9]{30,}",
        rb"/home/(?!user/|username/)[A-Za-z0-9_.-]+/",
        rb"\b(?:PetEnvCfg|PetAMPPPORunnerCfg|pet-amp|PET_CFG)\b",
        rb'"Frames"\s*:\s*\[',
    ]
    for path in filter(None, index):
        if forbidden(path):
            failures.add("index: " + path)
            continue
        content = git("show", ":" + path)
        if path != "scripts/check_publication.py" and any(re.search(pattern, content) for pattern in patterns):
            failures.add("review indexed content: " + path)
    # Inspect every reachable tree, including files deleted by later commits.
    for revision in git("rev-list", "--all").decode().splitlines():
        for path in git("ls-tree", "-r", "--name-only", revision).decode().splitlines():
            if forbidden(path):
                failures.add("history: " + path)
    for failure in sorted(failures):
        print(failure)
    if failures:
        print("BLOCKED: review/sanitize the index and inherited history before pushing.")
        return 1
    print("PASS: index and historical path checks passed; manually review the staged diff too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
