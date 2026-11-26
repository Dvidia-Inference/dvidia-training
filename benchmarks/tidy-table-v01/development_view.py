"""Render the first 80-mm lift on a development fixture from an external viewpoint."""
import argparse
from pathlib import Path

from mujoco.rendering.classic.renderer import Renderer

from dvidia_training.piper_action_capture import _png
from dvidia_training.tidy_table_native import NativeTable


class Illustration(NativeTable):
    output = None

    def _update_settled(self):
        super()._update_settled()
        if self.output is None or self.data.xpos[self.block_ids["A"], 2] < .16:
            return
        camera = self.mj.MjvCamera()
        camera.lookat[:] = [.24, -.01, .19]
        camera.azimuth, camera.elevation, camera.distance = 120., -24., .98
        global_view = self.model.vis.global_
        global_view.offwidth, global_view.offheight = 1000, 640
        with Renderer(self.model, 640, 1000) as renderer, self.output.open("xb") as stream:
            renderer.update_scene(self.data, camera=camera)
            stream.write(_png(renderer.render()))
        self.output = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()
    with Illustration(args.assets, {"A":"center", "B":"left", "C":"right"}) as table:
        table.output = args.png
        table.observe()
        result = table.move("A", "buffer")
        if table.output is not None or not result["verified"]:
            raise RuntimeError("Development lift was not reproduced")
