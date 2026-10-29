"""Build a collision-geometry MuJoCo model from the ChituMax URDF."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


def load_chitu_model(urdf_path: Path, timestep: float):
    """Import the URDF collision shapes. STL visuals are dropped."""
    source = ET.parse(urdf_path).getroot()
    for link in source.findall("link"):
        for visual in link.findall("visual"):
            link.remove(visual)
    extension = ET.SubElement(source, "mujoco")
    ET.SubElement(extension, "compiler", fusestatic="false", discardvisual="true")
    model = mujoco.MjModel.from_xml_string(ET.tostring(source, encoding="unicode"))

    import tempfile

    with tempfile.TemporaryDirectory(prefix="chitu-mujoco-") as directory:
        converted = Path(directory) / "chitu.xml"
        mujoco.mj_saveLastXML(str(converted), model)
        root = ET.parse(converted).getroot()

    compiler = root.find("compiler")
    compiler.set("inertiafromgeom", "false")
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(timestep))
    option.set("gravity", "0 0 -9.81")
    world = root.find("worldbody")
    base = world.find("body")
    ET.SubElement(base, "freejoint", name="floating_base")
    ET.SubElement(base, "site", name="imu", size="0.005")
    for geom in root.iter("geom"):
        if geom.get("type") == "cylinder":
            geom.set("type", "capsule")
        geom.set("rgba", "0.55 0.6 0.65 1")
        geom.set("friction", "1 0.005 0.0001")
    for joint in root.iter("joint"):
        joint.set("damping", "0")
        joint.set("frictionloss", "0.01")
        joint.set("armature", "0")
    ET.SubElement(world, "geom", name="ground", type="plane", size="20 20 0.1",
                  rgba="0.25 0.3 0.35 1", friction="1 0.005 0.0001")
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1")
    actuators = ET.SubElement(root, "actuator")
    for joint in base.iter("joint"):
        ET.SubElement(actuators, "motor", name=joint.get("name") + "_motor",
                      joint=joint.get("name"), gear="1")
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "gyro", name="imu_gyro", site="imu")
    ET.SubElement(sensors, "framequat", name="imu_quat", objtype="site", objname="imu")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
