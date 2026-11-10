extends SceneTree
## Run: godot --headless --path integrations/godot --script tests/test_coordinates.gd

const Workstation = preload("res://main.gd")


func _init() -> void:
	var checks := 0
	for pair in [
		[[1, 0, 0], Vector3.RIGHT], [[0, 1, 0], Vector3(0, 0, -1)],
		[[0, 0, 1], Vector3.UP], [[0.3, -0.2, 0.7], Vector3(0.3, 0.7, 0.2)],
	]:
		assert(Workstation.native_vector(pair[0]).is_equal_approx(pair[1]))
		checks += 1
	var converted := Workstation.native_basis([cos(PI / 4), 0, 0, sin(PI / 4)])
	assert((converted * Vector3.RIGHT).is_equal_approx(Vector3(0, 0, -1)))
	assert(is_equal_approx(converted.determinant(), 1.0))
	checks += 2
	# body(R * local + p) must equal C*body(R*local+p), with no double axis
	# conversion and no parent-body transformation of an already-world pose.
	var native_rotation := Basis(Quaternion(Vector3(0.3, 0.7, -0.2).normalized(), 0.81))
	var q := native_rotation.get_rotation_quaternion()
	var mapped_rotation := Workstation.native_basis([q.w, q.x, q.y, q.z])
	var local := Vector3(0.12, -0.03, 0.08)
	var position := Vector3(0.26, -0.14, 0.22)
	var native_result := native_rotation * local + position
	var godot_result := mapped_rotation * Workstation.native_vector([local.x, local.y, local.z]) + Workstation.native_vector([position.x, position.y, position.z])
	assert(godot_result.distance_to(Workstation.native_vector([native_result.x, native_result.y, native_result.z])) < 0.000001)
	checks += 1
	print(JSON.stringify({"schema": "dvidia.godot-coordinates-test/v1", "passed": checks}))
	quit(0)
