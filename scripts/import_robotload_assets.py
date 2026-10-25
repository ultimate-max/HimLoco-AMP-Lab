"""Restore ChituMax assets from the pinned RobotLoad commit without checking it out."""

import argparse
import io
from pathlib import Path
import subprocess
import tarfile


REVISION = "b4ae8a0ddc86094b240f01f1bfc87084164e7d56"
PATHS = (
    "datasets/chitu/mocap_motions",
    "resources/robots/chitu",
    "resources/friction/chitu_mgms_net_params.json",
)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "AMP_for_hardware")
    args = parser.parse_args()
    archive = subprocess.check_output(
        ["git", "-C", str(args.source), "archive", REVISION, *PATHS]
    )
    destination = root / "resources/robotload"
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()) or member.issym() or member.islnk():
                raise ValueError(f"Unsafe archive entry: {member.name}")
            if not member.isfile():
                continue
            contents = bundle.extractfile(member).read()
            if target.exists():
                if target.read_bytes() != contents:
                    raise FileExistsError(f"Refusing to overwrite modified asset: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
    print(f"RobotLoad assets ready: {destination} (commit {REVISION})")


if __name__ == "__main__":
    main()
