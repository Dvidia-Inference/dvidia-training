# What Tesla’s Reality View suggests for PiPER

Tesla’s November 21, 2026 introduction of Reality View showed a live driving scene rendered from a virtual camera above the vehicle. Earlier that year, Tesla’s CVPR exhibit gave visitors a steering wheel to navigate a generated world. The first demonstration changes where the viewer looks from. The second changes what the vehicle does and shows the resulting scene. For a gripper, the corresponding research question is whether a model can predict how an object will respond when two different actions are possible from the same grasp. [Tesla’s Reality View announcement](https://www.youtube.com/watch?v=eNBqkc7V1nU) · [Tesla’s CVPR exhibit](https://www.tesla.com/en_ca/event/tesla-x-cvpr-2026)

Our November 11 recorder study established a starting point. In three paired comparisons, the PiPER camera images, depth, commands and physics traces matched across both recorder implementations. Every control interval with a camera update exceeded the 20 ms target on the benchmark machine. The arm moved above a block throughout the study. Evaluating a prediction about grip requires episodes with actual contact and object motion. [Capture benchmark](../benchmarks/godot-capture-latency-v02/README.md)

## Proposed hold-or-lift experiment

1. Record a simulated PiPER closing its gripper on a block. Vary the starting pose, grasp offset and friction to include both secure and marginal grasps. Save the complete simulator state at the decision instant. Run two branches from that state with the same jaw command. One holds the tool still for 200 ms. The other commands a 20 mm upward test lift over the same period. Record actual tool travel for both branches.
2. Give the predictor the proposed command and the same timestamped history before the decision in both branches. Include available RGB-D estimates, their age, encoder readings, jaw position and prior commands. Keep occlusions marked as missing. Ask for block displacement at 200 ms and a probability of losing the grasp.
3. Score against the simulator’s recorded block and tool poses and contacts. Measure displacement error in millimetres. Define loss of grasp and its persistence threshold before collection, using relative block-to-gripper motion and contact history. Score the change in displacement between the two branches of each pair. That catches a predictor that always returns the most common outcome.
4. Keep both branches and every recording from a starting configuration in the same split. Freeze the held-out configurations and label rules before fitting. Compare with a stationary-block baseline and a matched predictor trained without the proposed command. Report missed slips, false alarms, the number of true losses, and paired uncertainty grouped by start. Measure observation age and decision latency on the target laptop alongside prediction error.

## First simulated hold and lift

On November 25 we ran the mechanics portion at one block position with seed 0. An authored controller approached the block and issued 35 closing commands. Within each jaw condition, both branches used the same 563-command prefix. The probe checked the compiled model, MuJoCo integration state, servo targets, task status and recorded native history for exact agreement at the decision. The receipt contains the encoder readings and previous command in one field and the physics measurements used for scoring in another. [Probe source](../scripts/probe-piper-action-branches.py) · [Machine-readable results](../benchmarks/piper-perception-latency-v03/action-branches.json)

| Jaw and action | Tool rise | Block rise | Both fingers in contact | Block supported by table |
|---|---:|---:|---:|---:|
| Closed, hold | −0.02 mm | −0.03 mm | 200/200 ticks | 200/200 ticks |
| Closed, lift | 10.89 mm | 10.82 mm | 200/200 ticks | 20/200 ticks |
| Underclosed, hold | −0.001 mm | 0 mm | 0/200 ticks | 200/200 ticks |
| Underclosed, lift | 11.74 mm | 0 mm | 0/200 ticks | 200/200 ticks |

The command requested 20 mm of vertical tool motion over 200 ms. The two lift branches also drifted horizontally by 5.89 and 6.44 mm, respectively. A predictor evaluated against the commanded height would therefore receive the wrong target even for the closed grasp. The 32 mm single-finger target produced a failed acquisition, with the block supported throughout. A slip test still needs an initially acquired block that subsequently moves relative to the gripping fingers. This probe measured simulated mechanics. Predictor accuracy on held-out grasps and physical transfer remain open.

To reproduce, run the script from this checkout with the arm simulation dependencies and the verified PiPER assets. Choose a new output filename.

```sh
PYTHONPATH=src python scripts/probe-piper-action-branches.py --assets /path/to/assets --output action-branches.json
```
