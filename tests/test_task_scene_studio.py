import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dvidia_training import task_scene
from dvidia_training.task_scene_studio import SceneServer, decode


class StudioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.calls = []
        def author(recipe, output):
            output.mkdir()
            (output/"recipe.json").write_text(json.dumps(recipe))
            return {"scope": "test authoring fixture; actual USD is tested separately"}
        self.author_patch = patch("dvidia_training.task_scene.write_scene", side_effect=author)
        self.author_patch.start()
        self.addCleanup(self.author_patch.stop)
        def runner(recipe, assets, output, **kwargs):
            self.calls.append(task_scene.load_recipe(recipe))
            output.mkdir()
            (output/"index.json").write_text('{"cases": []}')
            return {"scope": "test stub"}
        self.server = SceneServer(("127.0.0.1", 0), Path(self.tmp.name)/"runs", Path(self.tmp.name)/"assets",
                                  runner=runner, robot_preview=False)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:"+str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.author_patch.stop()
        self.tmp.cleanup()

    def request(self, path, data=None, **headers):
        request = Request(self.base+path, data=None if data is None else json.dumps(data).encode(),
                          headers={"Content-Type": "application/json", **headers})
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    def test_local_recipe_round_trip_and_retained_replay(self):
        config = self.request("/api/config")
        self.assertTrue(config["local"])
        result = self.request("/api/simulate", {"recipe": config["recipe"]})
        self.assertEqual(self.request(result["index_url"]), {"cases": []})
        self.assertEqual(self.calls, [config["recipe"]])
        with self.assertRaises(HTTPError):
            self.request(result["index_url"].replace("recording/index.json", "../../secret"))

    def test_cross_origin_and_invalid_recipe_never_execute(self):
        for payload, headers in (({"recipe": task_scene.default_recipe()}, {"Origin": "https://example.com"}),
                                 ({"recipe": {}}, {}), ({"recipe": task_scene.default_recipe(), "path": "/tmp"}, {})):
            with self.assertRaises(HTTPError):
                self.request("/api/simulate", payload, **headers)
        self.assertEqual(self.calls, [])

    def test_busy_worker_never_starts_second_run(self):
        self.server.worker_lock.acquire()
        try:
            with self.assertRaises(HTTPError):
                self.request("/api/simulate", {"recipe": task_scene.default_recipe()})
        finally:
            self.server.worker_lock.release()
        self.assertEqual(self.calls, [])

    def test_ambiguous_json_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.assertRaises(ValueError):
                decode(raw)

    def test_wrong_host_transfer_encoding_and_oversized_body_do_not_execute(self):
        for headers in ({"Host": "evil.invalid"}, {"Transfer-Encoding": "chunked"},
                        {"Content-Length": "32769"}, {"Content-Type": "text/plain"}):
            with self.subTest(headers=headers), self.assertRaises(HTTPError):
                self.request("/api/simulate", {"recipe": task_scene.default_recipe()}, **headers)
        self.assertEqual(self.calls, [])

    def test_symlink_artifacts_are_never_served(self):
        result = self.request("/api/simulate", {"recipe": task_scene.default_recipe()})
        run_id = result["index_url"].split("/")[2]
        secret = Path(self.tmp.name)/"outside.json"
        secret.write_text('{"secret": true}')
        (self.server.directory/run_id/"recording/link.json").symlink_to(secret)
        with self.assertRaises(HTTPError):
            self.request("/runs/"+run_id+"/recording/link.json")

    def test_failed_native_attempt_is_retained_and_worker_unlocks(self):
        def failure(*args, **kwargs):
            raise RuntimeError("simulated native refusal")
        self.server.runner = failure
        with self.assertRaises(HTTPError) as caught:
            self.request("/api/simulate", {"recipe": task_scene.default_recipe()})
        self.assertEqual(caught.exception.code, 500)
        self.assertEqual(len(self.server.runs), 1)
        run_id = next(iter(self.server.runs))
        error = self.request("/runs/"+run_id+"/error.json")
        self.assertEqual(error["status"], "failed")
        self.assertIn("native refusal", error["error"])
        self.assertTrue((self.server.directory/run_id/"input/recipe.json").is_file())
        self.assertFalse(self.server.worker_lock.locked())


if __name__ == "__main__":
    unittest.main()
