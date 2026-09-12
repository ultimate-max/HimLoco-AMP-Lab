import mujoco
import mujoco.viewer
import time

xml_string = """
<mujoco>
    <worldbody>
        <light name="top" pos="0 0 1"/>
        <geom name="floor" type="plane" size="1 1 0.1"/>
        <body name="box" pos="0 0 0.5">
            <joint type="free"/>
            <geom type="box" size="0.1 0.2 0.3" rgba="1 0 0 1"/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

print("Staring Viewer......")
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)