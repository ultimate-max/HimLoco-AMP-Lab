# ChituMax RobotLoad assets

Source: local AMP_for_hardware repository, origin/RobotLoad,
commit b4ae8a0ddc86094b240f01f1bfc87084164e7d56.

The resources directory contains the original chitu_max URDF, its STL meshes,
and identified MGMS friction parameters. The payload and foot fixed links must
remain separate in the Isaac Lab articulation.

The 62 reference motion files in datasets/chitu/mocap_motions are local data,
excluded from Git consistently with the project's dataset publication policy.
They are copied without modification; preprocessing happens in AMPLoader.
Restore them from the pinned source commit with:

```bash
python scripts/import_robotload_assets.py --source /path/to/AMP_for_hardware
python scripts/check_chitu_amp.py
```

The importer never switches branches and refuses to overwrite changed assets.
