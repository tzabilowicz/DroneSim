## Adding Target to Scene
1. Run `py clone_assets.py`
2. Place the target.xml file in the `\mujoco_menagerie\skydio_x2\` directory after the clone script is run.
3. Update the `scene.xml` to add `<include file="target.xml"/>`
4. Change the scene rendering by updating `<statistic center="0 0 5" extent="100" meansize=".2"/>`