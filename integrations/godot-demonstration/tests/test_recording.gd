extends SceneTree
## Recorder contract assertions without a socket or an imported robot scene.

const Demonstration = preload("res://main.gd")


func episode(status: String = "ready", steps: int = 0) -> Dictionary:
	return {"status": status, "steps": steps, "max_steps": 3, "source_kind": "simulation_scripted_input", "output_directory": "/tmp/dvidia-test/episode-001"}


func _init() -> void:
	var checks := 0
	assert(Demonstration.SCHEMA == "dvidia.godot-demonstration/v1")
	assert(Demonstration.valid_recording(episode()))
	checks += 2
	for invalid in [null, {}, {"status": "training"}, episode("ready", -1), episode("recording", 4)]:
		assert(not Demonstration.valid_recording(invalid))
		checks += 1
	for replacement in [{"steps": 0.5}, {"steps": NAN}, {"max_steps": 0}, {"source_kind": "human"}, {"output_directory": "relative/episode"}]:
		var invalid := episode()
		invalid.merge(replacement, true)
		assert(not Demonstration.valid_recording(invalid))
		checks += 1
	var ready := episode()
	var recording := episode("recording")
	var first := episode("recording", 1)
	var paused := episode("paused", 1)
	var finished := episode("finished", 1)
	for transition in [[{}, ready, "hello"], [ready, recording, "begin"], [recording, first, "step"], [first, paused, "pause"], [paused, first, "resume"], [paused, finished, "finish"], [finished, finished, "finish"], [episode("recording", 2), episode("finished", 3), "step"]]:
		assert(Demonstration.recording_transition_error(transition[0], transition[1], transition[2]).is_empty())
		checks += 1
	for transition in [[ready, first, "begin"], [first, episode("paused", 2), "pause"], [first, finished, "step"], [first, ready, "reset"]]:
		assert(not Demonstration.recording_transition_error(transition[0], transition[1], transition[2]).is_empty())
		checks += 1
	var new_episode := episode()
	new_episode.output_directory = "/tmp/dvidia-test/episode-002"
	assert(Demonstration.recording_transition_error(finished, new_episode, "reset").is_empty())
	assert(not Demonstration.recording_transition_error(first, new_episode, "step").is_empty())
	checks += 2
	# Programmatic Home/smoke commands must use the slider's clamped value.
	var client := Demonstration.new()
	var slider := HSlider.new()
	slider.min_value = 0.1
	slider.max_value = 0.5
	slider.step = 0.001
	client.sliders.append(slider)
	client.value_labels.append(Label.new())
	client.marker = MeshInstance3D.new()
	client._set_slider(0, 0.55)
	assert(is_equal_approx(client.target[0], 0.5))
	client._set_slider(0, -0.01)
	assert(is_equal_approx(client.target[0], 0.1))
	assert(is_equal_approx(client.marker.position.x, 0.1))
	checks += 3
	slider.free()
	client.value_labels[0].free()
	client.marker.free()
	client.free()
	print(JSON.stringify({"schema": "dvidia.godot-demonstration-recording-test/v1", "passed": checks}))
	quit(0)
