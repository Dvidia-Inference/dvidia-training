# Image-goal verification with Jev

DVIDIA research · protocol v0.1 · 18 October 2026

This benchmark asks whether an image-goal evaluator can distinguish a finished
placement from a scene that merely looks finished. It runs in our native PiPER
workstation and uses **Jev 1.14.0 by TypeSafe** as an optional hosted evaluator.
It does not train a robot, produce actuator commands, or install Jev locally.

## What the experiment measures

Four authored layouts each contain eight conditions: completed placement,
missing target, wrong object, wrong position, wrong orientation, contact without
release, motion with an identical rendered pose, and an occluded target.
The resulting 32 cases are evaluated twice: current/goal RGB images alone,
and the same images with privileged simulator measurements. These are correlated
diagnostic cases, not 32 independent real-world tasks or a held-out deployment
success estimate.

The physical rubric requires the intended green shoe within 25 mm in the table
plane and 15 degrees in yaw, supported by the table, released, and stationary
for 200 ms. Exact force and speed thresholds are in the frozen protocol. The
ground truth is computed from native state and contact observations every 2 ms.
The telemetry condition supplies numeric observations and extrema over the
complete native window, without a completion label. Object identity and pose
errors are privileged simulator information, not a validated vision estimator.

The shoe is an explicitly declared 30 g rigid proxy with half the catalog shoe
width. The PiPER drives are unchanged. This enables a real finger-contact
negative control; it is not a demonstration of full-size shoe grasping,
deformation or hardware transfer. State fixtures are initialized deliberately;
the moving control changes native velocity instantaneously without claiming a
physically applied impulse or successful manipulation trajectory.

The completed and moving fixtures have byte-identical images in each layout.
RGB alone therefore cannot distinguish their physical completion labels.
The correct evidence-aware response is **unknown** when hidden velocity,
support, release or dwell cannot be established. We separately report physical
false completion, observability-aware accuracy, justified abstention, decision
coverage, presence errors on absent targets, and probability Brier scores.
An always-abstaining system has zero false completion but zero completion
coverage; neither number should be presented alone.

## Reproduce locally

Use Python 3.12 or newer, MuJoCo 3.15.0 and the repository's pinned dependencies.
Native fixture rendering needs a working OpenGL context. On a headless Linux
server, configure EGL or OSMesa for your environment. A discrete GPU is not a
requirement for the simulator physics, but rendering support is required.
Measured time on one host is not a minimum hardware specification.

```sh
python -m pip install '.[arm]'
dvidia-robot-sim fetch piper --output robot-assets
dvidia-image-goal generate --assets robot-assets --output runs/image-goal-fixtures
dvidia-image-goal freeze --fixtures runs/image-goal-fixtures --source-commit "$(git rev-parse HEAD)"
dvidia-image-goal evaluate --fixtures runs/image-goal-fixtures --output runs/image-goal-evaluation
```

Configure `TYPESAFE_API_KEY` securely in the process environment before the final
command. Do not commit a key, put it in a shell argument, or include it in a
benchmark artifact. On macOS, a Keychain item can be selected by nonsecret names:

```sh
dvidia-image-goal evaluate --fixtures runs/image-goal-fixtures \
  --output runs/image-goal-evaluation \
  --keychain-service dvidia-typesafe --keychain-account dvidia-research
```

Without credentials, the runner makes no network requests and writes explicit
unrun trials. Accuracy, latency and cost remain unmeasured. Use a fresh output
directory for each evaluation; existing evidence cannot be overwritten.

The evaluation seals images, labels, prompts, implementation hashes and the full
paired trial inventory before calls. It prevalidates every request, executes
at most 64 requests serially with no automatic retries, and retains all results
and errors. Each request uses a 20-second socket-operation timeout; this is not
a hard wall-clock or robotics deadline. A checkpoint always preserves the full
scheduled inventory, including unattempted trials after interruption.

Latency includes request preparation, image upload, network/service delay and
response parsing from preloaded PNGs. It excludes fixture loading and rendering,
camera capture, actuation and physical task execution. Report successful-request
latency separately from all attempts. p99 from a small cohort is descriptive,
not a reliable tail-latency guarantee. Missing responses never become correct
negative predictions, and unknown billed usage remains explicitly unknown.

## Jev integration and licensing

The client uses the documented multipart API, the experimental image header,
two Noul probabilities and one three-way Choice verdict. Presence thresholds
are fixed at 0.2 and 0.8 before evaluation. There is no bounding-box grasp path:
TypeSafe's Box primitive alone has no absent-object answer.

Our code is MIT under the repository license. PiPER assets retain their pinned
upstream notices. Jev is a separately operated hosted service; no Jev weights
are included. Its current service terms restrict model imitation/distillation.
This experiment evaluates decisions and does not use service outputs to train
a replacement model. Only authored simulator images and observations are sent.

Primary references, checked 18 October 2026:

- [Jev 1.14 release](https://typesafe.ai/blog/jev-1-14)
- [Experimental image API and limits](https://docs.typesafe.ai/images)
- [Noul probabilities](https://docs.typesafe.ai/primitives/noul)
- [Choice decisions](https://docs.typesafe.ai/primitives/choice)
- [Models and pricing](https://docs.typesafe.ai/models)
- [Service terms](https://typesafe.ai/legal/mca)

Published model results must identify the frozen source, case manifest, protocol,
request inventory and returned model version. A valid local fixture run or an
offline client test is not a measured Jev result.
