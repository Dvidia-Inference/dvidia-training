extends Node3D
## Local renderer, never a robot physics authority. MIT, DVIDIA contributors.

const SCHEMA := "dvidia.godot-bridge/v1"
const MAX_FILE := 64 * 1024 * 1024
const MAX_PACKET := 1024 * 1024
const INK := Color("25322e")
const MUTED := Color("728079")
const GREEN := Color("4e7562")
const PAPER := Color("f8f8f2")

var peer := StreamPeerTCP.new()
var scene_data: Dictionary = {}
var body_nodes: Array[Node3D] = []
var mesh_cache: Dictionary = {}
var incoming := PackedByteArray()
var connected := false
var hello_sent := false
var pending := false
var pending_kind := ""
var paused := false
var finished := false
var errors: Array[String] = []
var sequence := 1
var replies := 0
var resets := 0
var state_time := 0.0
var compute_total_ms := 0.0
var last_compute_ms := 0.0
var last_request_us := 0
var next_step_due_us := 0
var started_us := 0
var connected_us := 0
var frame_count := 0
var target := [0.36, -0.06, 0.27]
var jaw := 0.035
var sliders: Array[HSlider] = []
var value_labels: Array[Label] = []
var status_label: Label
var time_label: Label
var compute_label: Label
var pause_button: Button
var camera: Camera3D
var marker: MeshInstance3D
var camera_yaw := -0.63
var camera_pitch := 0.42
var camera_distance := 1.65
var orbiting := false
var focus := Vector3(0.26, 0.12, 0.0)
var scene_dir := ""
var port := 8430
var smoke_steps := 0
var smoke_output := ""
var screenshot_path := ""
var initial_sim_time := 0.0
var initial_state: Dictionary = {}
var smoke_trace: Array[Dictionary] = []
var pending_request: Dictionary = {}
var reset_requested := false
var scene_integrity_sha256 := ""
var verified_files: Dictionary = {}
var ready_us := 0
var completed_us := 0
var frame_intervals_ms: Array[float] = []
var rpc_roundtrip_ms: Array[float] = []


func _ready() -> void:
	PhysicsServer3D.set_active(false)
	PhysicsServer2D.set_active(false)
	started_us = Time.get_ticks_usec()
	_parse_args()
	_build_world()
	_build_ui()
	if not errors.is_empty():
		_fail(errors[0])
		return
	if scene_dir.is_empty():
		_fail("Choose an exported scene: --scene /absolute/path --port 8430")
		return
	if not _load_scene():
		return
	status_label.text = "Connecting to local MuJoCo…"
	var result := peer.connect_to_host("127.0.0.1", port)
	if result != OK:
		_fail("Cannot connect to local MuJoCo (%s)." % result)
	connected_us = Time.get_ticks_usec()


func _parse_args() -> void:
	var args := OS.get_cmdline_user_args()
	var index := 0
	while index < args.size():
		var flag := args[index]
		if flag not in ["--scene", "--port", "--smoke-steps", "--smoke-output", "--screenshot"] or index + 1 >= args.size():
			errors.append("Unknown or incomplete argument: " + flag)
			return
		var value := args[index + 1]
		match flag:
			"--scene": scene_dir = value
			"--port": port = int(value)
			"--smoke-steps": smoke_steps = int(value)
			"--smoke-output": smoke_output = value
			"--screenshot": screenshot_path = value
		index += 2
	if port < 1024 or port > 65535 or smoke_steps < 0 or smoke_steps > 100000:
		errors.append("Port or smoke step count is outside bounds.")
	if (not smoke_output.is_empty() and not smoke_output.is_absolute_path()) or (not screenshot_path.is_empty() and not screenshot_path.is_absolute_path()):
		errors.append("Smoke output and screenshot paths must be absolute.")
	if (not smoke_output.is_empty() and FileAccess.file_exists(smoke_output)) or (not screenshot_path.is_empty() and FileAccess.file_exists(screenshot_path)):
		errors.append("Smoke output and screenshot paths must be fresh files.")


func _build_world() -> void:
	var environment_node := WorldEnvironment.new()
	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = Color("e9ebe2")
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.ambient_light_color = Color("f1f5e9")
	# Compatibility rendering is SDR. Keep the light budget below clipping;
	# bright HDR-style lighting washes out native RGBA materials in this path.
	environment.ambient_light_energy = 0.28
	environment.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	environment_node.environment = environment
	add_child(environment_node)
	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-52, -38, 0)
	light.light_energy = 0.68
	light.shadow_enabled = true
	light.directional_shadow_max_distance = 5.0
	add_child(light)
	var fill := DirectionalLight3D.new()
	fill.rotation_degrees = Vector3(-32, 130, 0)
	fill.light_energy = 0.16
	add_child(fill)
	camera = Camera3D.new()
	camera.fov = 41
	camera.near = 0.005
	camera.far = 30
	add_child(camera)
	camera.make_current()
	_position_camera()
	marker = MeshInstance3D.new()
	var sphere := SphereMesh.new()
	sphere.radius = 0.008
	sphere.height = 0.016
	marker.mesh = sphere
	var material := StandardMaterial3D.new()
	material.albedo_color = Color("73a27e")
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	marker.material_override = material
	add_child(marker)
	marker.position = native_vector(target)


func _style(background: Color, border: Color = Color.TRANSPARENT) -> StyleBoxFlat:
	var box := StyleBoxFlat.new()
	box.bg_color = background
	box.border_color = border
	box.set_border_width_all(1)
	box.set_corner_radius_all(8)
	box.content_margin_left = 16
	box.content_margin_right = 16
	box.content_margin_top = 11
	box.content_margin_bottom = 11
	return box


func _label(text: String, size: int = 16, color: Color = INK) -> Label:
	var label := Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", size)
	label.add_theme_color_override("font_color", color)
	return label


func _button(text: String, action: Callable) -> Button:
	var button := Button.new()
	button.text = text
	button.custom_minimum_size.y = 42
	button.add_theme_font_size_override("font_size", 15)
	button.add_theme_color_override("font_color", INK)
	button.add_theme_color_override("font_hover_color", INK)
	button.add_theme_stylebox_override("normal", _style(Color("eff1e9"), Color("d8ded1")))
	button.add_theme_stylebox_override("hover", _style(Color("e1e8dc")))
	button.add_theme_stylebox_override("pressed", _style(Color("d3dfcd")))
	button.pressed.connect(action)
	return button


func _build_ui() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	var root := Control.new()
	root.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	layer.add_child(root)
	var header := PanelContainer.new()
	header.set_anchors_and_offsets_preset(Control.PRESET_TOP_WIDE)
	header.offset_bottom = 72
	header.add_theme_stylebox_override("panel", _style(PAPER, Color("d9ded2")))
	root.add_child(header)
	var top := HBoxContainer.new()
	top.add_theme_constant_override("separation", 18)
	header.add_child(top)
	top.add_child(_label("DVIDIA", 23))
	top.add_child(_label("/  Manipulation lab", 16, MUTED))
	var spacer := Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	top.add_child(spacer)
	top.add_child(_label("PHYSICS  MuJoCo    ·    LOCAL COMPUTE", 13, GREEN))
	var panel := PanelContainer.new()
	panel.set_anchors_and_offsets_preset(Control.PRESET_TOP_RIGHT)
	panel.offset_left = -332
	panel.offset_right = -22
	panel.offset_top = 96
	panel.offset_bottom = 645
	panel.add_theme_stylebox_override("panel", _style(PAPER, Color("d9ded2")))
	root.add_child(panel)
	var column := VBoxContainer.new()
	column.add_theme_constant_override("separation", 15)
	panel.add_child(column)
	column.add_child(_label("PiPER workstation", 22))
	column.add_child(_label("Move the target. Watch the contact.", 14, MUTED))
	status_label = _label("Opening scene…", 14, GREEN)
	status_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	status_label.custom_minimum_size = Vector2(270, 38)
	column.add_child(status_label)
	for index in range(4):
		var row := HBoxContainer.new()
		var names := ["Target X", "Target Y", "Target Z", "Gripper target"]
		var caption := _label(names[index], 14)
		caption.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		row.add_child(caption)
		var value_label := _label("", 14, MUTED)
		row.add_child(value_label)
		value_labels.append(value_label)
		column.add_child(row)
		var slider := HSlider.new()
		slider.custom_minimum_size.y = 18
		slider.step = 0.001
		slider.editable = false
		slider.value_changed.connect(_slider_changed.bind(index))
		sliders.append(slider)
		column.add_child(slider)
	var buttons := HBoxContainer.new()
	buttons.add_theme_constant_override("separation", 8)
	pause_button = _button("Pause", _toggle_pause)
	pause_button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	buttons.add_child(pause_button)
	buttons.add_child(_button("Home", _home))
	buttons.add_child(_button("Reset", _reset))
	column.add_child(buttons)
	time_label = _label("Simulation  0.00 s", 15)
	compute_label = _label("Native step  — ms", 13, MUTED)
	column.add_child(time_label)
	column.add_child(compute_label)
	var note := _label("Synthetic rigid shoe · Research prototype\nContact and material properties are uncalibrated.", 12, MUTED)
	note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	column.add_child(note)
	var hints := _label("Drag empty space to orbit   ·   Scroll to zoom\nGreen point: commanded target", 14, MUTED)
	hints.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_LEFT)
	hints.offset_left = 28
	hints.offset_top = -63
	hints.offset_bottom = -18
	root.add_child(hints)


static func native_vector(value: Array) -> Vector3:
	return Vector3(float(value[0]), float(value[2]), -float(value[1]))


static func native_basis(value: Array) -> Basis:
	# A proper coordinate rotation C: [x,y,z] -> [x,z,-y]. Both local
	# mesh coordinates and every transform are mapped, so R' = C R C^-1.
	return Basis(Quaternion(float(value[1]), float(value[3]), -float(value[2]), float(value[0])))


func _valid_vector(value: Variant, length: int) -> bool:
	if not value is Array or value.size() != length:
		return false
	for number in value:
		if not (number is int or number is float) or not is_finite(float(number)):
			return false
	return true


func _valid_quaternion(value: Variant) -> bool:
	if not _valid_vector(value, 4):
		return false
	var norm_squared := 0.0
	for number in value:
		norm_squared += float(number) * float(number)
	return absf(norm_squared - 1.0) < 0.00001


func _load_scene() -> bool:
	if not _verify_scene_files():
		return false
	var decoded: Variant = JSON.parse_string(verified_files["scene.json"].get_string_from_utf8())
	if not decoded is Dictionary or decoded.get("schema") != "dvidia.asset-preview-model/v1" or not decoded.has("bridge") or not decoded.has("bodies") or not decoded.has("meshes") or not decoded.has("geoms"):
		_fail("Scene metadata is incomplete.")
		return false
	scene_data = decoded
	var bridge: Dictionary = scene_data.bridge
	if bridge.get("schema") != SCHEMA or not str(bridge.get("model_sha256", "")).is_valid_hex_number(false) or str(bridge.get("model_sha256", "")).length() != 64:
		_fail("Scene bridge identity is invalid.")
		return false
	if float(bridge.get("physics_dt", -1)) != 0.002 or float(bridge.get("control_dt", -1)) != 0.02 or scene_data.get("source", {}).get("compiled_model_sha256") != bridge.model_sha256:
		_fail("Scene model source or pinned simulation clocks differ.")
		return false
	var info: Dictionary = scene_data.get("geometry_binary", {})
	if info.get("href") != "geometry.bin" or info.get("compression") != "none" or int(info.get("bytes", -1)) < 0 or int(info.get("bytes", MAX_FILE + 1)) > MAX_FILE:
		_fail("Scene geometry framing is invalid.")
		return false
	var bytes: PackedByteArray = verified_files["geometry.bin"]
	if bytes.size() != int(info.bytes):
		_fail("Geometry file is missing or has changed size.")
		return false
	var hash := HashingContext.new()
	hash.start(HashingContext.HASH_SHA256)
	hash.update(bytes)
	if hash.finish().hex_encode() != info.get("sha256"):
		_fail("Geometry checksum does not match the exported scene.")
		return false
	if scene_data.bodies.size() < 1 or scene_data.bodies.size() > 128 or scene_data.meshes.size() > 128 or scene_data.geoms.size() > 512:
		_fail("Scene inventory exceeds bounds.")
		return false
	for body in scene_data.bodies:
		if int(body.id) != body_nodes.size():
			_fail("Body IDs must be contiguous native IDs.")
			return false
		var node := Node3D.new()
		node.name = str(body.name)
		add_child(node)
		body_nodes.append(node)
	for record in scene_data.meshes:
		var mesh := _native_mesh(record, bytes)
		if mesh == null:
			return false
		mesh_cache[int(record.id)] = mesh
	for geom in scene_data.geoms:
		if not _add_geom(geom):
			return false
	if not _valid_vector(bridge.get("home_target"), 3) or not bridge.get("bounds") is Array or bridge.bounds.size() != 3:
		_fail("Target declaration is invalid.")
		return false
	target = bridge.home_target.duplicate()
	jaw = float(bridge.gripper_initial)
	for index in range(4):
		if index < 3:
			sliders[index].min_value = float(bridge.bounds[index][0])
			sliders[index].max_value = float(bridge.bounds[index][1])
		else:
			sliders[index].min_value = 0
			sliders[index].max_value = 0.035
		_set_slider(index, float(target[index]) if index < 3 else jaw)
	verified_files.clear()
	return true


func _verify_scene_files() -> bool:
	var seal_file := FileAccess.open(scene_dir.path_join("scene-integrity.json"), FileAccess.READ)
	if seal_file == null or seal_file.get_length() > 65536:
		_fail("Cannot read a bounded scene-integrity.json.")
		return false
	var seal_bytes := seal_file.get_buffer(seal_file.get_length())
	scene_integrity_sha256 = _sha256(seal_bytes)
	var seal: Variant = JSON.parse_string(seal_bytes.get_string_from_utf8())
	if not seal is Dictionary or seal.get("schema") != "dvidia.godot-scene-integrity/v1" or not seal.get("files") is Dictionary:
		_fail("Scene integrity manifest is invalid.")
		return false
	var expected := ["scene.json", "geometry.bin", "LICENSE", "licenses/agilex_piper/LICENSE", "licenses/agilex_piper/README.md", "README.md"]
	if seal.files.size() != expected.size():
		_fail("Scene integrity inventory differs from the pinned PiPER export.")
		return false
	var total := 0
	for name in expected:
		var record: Variant = seal.files.get(name)
		if not record is Dictionary or not record.get("bytes") is float and not record.get("bytes") is int:
			_fail("Scene integrity file record is invalid.")
			return false
		var size := int(record.bytes)
		var limit := MAX_FILE if name == "geometry.bin" else (12 * 1024 * 1024 if name == "scene.json" else 65536)
		var file := FileAccess.open(scene_dir.path_join(name), FileAccess.READ)
		if size < 0 or size > limit or file == null or file.get_length() != size:
			_fail("Missing or size-mismatched scene artifact: " + name)
			return false
		var payload := file.get_buffer(size)
		if payload.size() != size or _sha256(payload) != record.get("sha256"):
			_fail("Scene artifact checksum mismatch: " + name)
			return false
		total += size
		if total > 96 * 1024 * 1024:
			_fail("Scene artifact total exceeds bounds.")
			return false
		if name in ["scene.json", "geometry.bin"]:
			verified_files[name] = payload
	var metadata: Variant = JSON.parse_string(verified_files["scene.json"].get_string_from_utf8())
	if not metadata is Dictionary or not metadata.get("bridge") is Dictionary or metadata.bridge.get("model_sha256") != seal.get("model_sha256"):
		_fail("Scene model identity differs from its integrity manifest.")
		return false
	return true


func _sha256(bytes: PackedByteArray) -> String:
	var hash := HashingContext.new()
	hash.start(HashingContext.HASH_SHA256)
	hash.update(bytes)
	return hash.finish().hex_encode()


func _native_mesh(record: Dictionary, bytes: PackedByteArray) -> ArrayMesh:
	var count := int(record.vertex_count)
	var indices := int(record.index_count)
	var vertex_offset := int(record.position_byte_offset)
	var index_offset := int(record.index_byte_offset)
	if count < 1 or indices < 3 or indices % 3 != 0 or vertex_offset < 0 or index_offset < 0 or vertex_offset + count * 12 > bytes.size() or index_offset + indices * 4 > bytes.size():
		_fail("Native mesh byte range is invalid.")
		return null
	var points := PackedVector3Array()
	points.resize(count)
	for index in range(count):
		var offset := vertex_offset + index * 12
		var value := [bytes.decode_float(offset), bytes.decode_float(offset + 4), bytes.decode_float(offset + 8)]
		if not _valid_vector(value, 3):
			_fail("Native mesh coordinate is nonfinite.")
			return null
		points[index] = native_vector(value)
	var triangle_indices := PackedInt32Array()
	triangle_indices.resize(indices)
	for index in range(0, indices, 3):
		# MuJoCo triangle winding is counterclockwise; Godot fronts are clockwise.
		for corner in range(3):
			var source_corner: int = [0, 2, 1][corner]
			var vertex := int(bytes.decode_u32(index_offset + (index + source_corner) * 4))
			if vertex < 0 or vertex >= count:
				_fail("Native triangle index is outside the mesh.")
				return null
			triangle_indices[index + corner] = vertex
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = points
	arrays[Mesh.ARRAY_INDEX] = triangle_indices
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	var surface := SurfaceTool.new()
	surface.create_from(mesh, 0)
	surface.generate_normals()
	return surface.commit()


func _add_geom(geom: Dictionary) -> bool:
	if not _valid_vector(geom.get("position"), 3) or not _valid_quaternion(geom.get("quaternion_wxyz")) or not _valid_vector(geom.get("size"), 3) or not _valid_vector(geom.get("rgba"), 4):
		_fail("Native geometry transform is invalid.")
		return false
	var body_id := int(geom.get("body_id", -1))
	if body_id < 0 or body_id >= body_nodes.size():
		_fail("Geometry references an unknown body.")
		return false
	if float(geom.rgba[3]) <= 0.0 or int(geom.group) > 2:
		return true
	var size: Array = geom.size
	var mesh: Mesh
	var scale_value := Vector3.ONE
	match geom.type:
		"mesh": mesh = mesh_cache.get(int(geom.mesh_id))
		"box":
			var box := BoxMesh.new()
			box.size = Vector3(2 * size[0], 2 * size[2], 2 * size[1])
			mesh = box
		"sphere", "ellipsoid":
			var sphere := SphereMesh.new()
			sphere.radius = 1.0 if geom.type == "ellipsoid" else float(size[0])
			sphere.height = 2 * sphere.radius
			if geom.type == "ellipsoid":
				scale_value = Vector3(size[0], size[2], size[1])
			mesh = sphere
		"capsule":
			var capsule := CapsuleMesh.new()
			capsule.radius = size[0]
			capsule.height = 2 * (size[0] + size[1])
			mesh = capsule
		"cylinder":
			var cylinder := CylinderMesh.new()
			cylinder.top_radius = size[0]
			cylinder.bottom_radius = size[0]
			cylinder.height = 2 * size[1]
			mesh = cylinder
		"plane":
			var plane := PlaneMesh.new()
			plane.size = Vector2(maxf(1, 2 * size[0]), maxf(1, 2 * size[1]))
			mesh = plane
	if mesh == null:
		_fail("Unsupported or missing native geometry: " + str(geom.type))
		return false
	var instance := MeshInstance3D.new()
	instance.name = str(geom.name)
	instance.mesh = mesh
	var rgba: Array = geom.rgba
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(rgba[0], rgba[1], rgba[2], rgba[3]).linear_to_srgb()
	material.roughness = 0.7
	material.metallic = 0.12 if geom.type == "mesh" else 0.0
	material.cull_mode = BaseMaterial3D.CULL_DISABLED
	if float(rgba[3]) < 1:
		material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	instance.material_override = material
	instance.transform = Transform3D(native_basis(geom.quaternion_wxyz).scaled_local(scale_value), native_vector(geom.position))
	body_nodes[body_id].add_child(instance)
	return true


func _process(delta: float) -> void:
	frame_count += 1
	if finished or not errors.is_empty():
		return
	if connected and frame_intervals_ms.size() < 10000:
		frame_intervals_ms.append(delta * 1000.0)
	peer.poll()
	var status := peer.get_status()
	if status == StreamPeerTCP.STATUS_CONNECTED:
		if not hello_sent:
			hello_sent = true
			_send({"type": "hello", "schema": SCHEMA, "model_sha256": scene_data.bridge.model_sha256})
			pending_kind = "hello"
		var available := peer.get_available_bytes()
		if available > 0:
			if incoming.size() + available > MAX_PACKET:
				_fail("Bridge packet exceeds bounds.")
				return
			var received := peer.get_data(available)
			if received[0] != OK:
				_fail("Local bridge read failed.")
				return
			incoming.append_array(received[1])
			while incoming.find(10) >= 0:
				var end := incoming.find(10)
				var message: Variant = JSON.parse_string(incoming.slice(0, end).get_string_from_utf8())
				incoming = incoming.slice(end + 1)
				if not message is Dictionary:
					_fail("Bridge returned invalid JSON.")
					return
				_receive(message)
				if finished or not errors.is_empty():
					return
		if connected and not pending:
			if reset_requested:
				reset_requested = false
				_home()
				pending_kind = "reset"
				_send({"type": "reset", "sequence": sequence})
			elif paused and Time.get_ticks_usec() - last_request_us >= 5000000:
				pending_kind = "ping"
				_send({"type": "ping"})
			elif not paused and Time.get_ticks_usec() >= next_step_due_us:
				if smoke_steps > 0:
					_smoke_target()
				# Preserve a 50 Hz deadline across render-frame quantization. There
				# is never more than one outstanding step, and overdue responses
				# discard the deadline rather than creating catch-up commands.
				next_step_due_us += 20000
				pending_kind = "step"
				_send({"type": "step", "sequence": sequence, "target": target, "jaw": jaw})
	elif status == StreamPeerTCP.STATUS_ERROR or hello_sent:
		_fail("MuJoCo disconnected. Simulation stopped.")
		return
	if (pending and Time.get_ticks_usec() - last_request_us > 10000000) or (not hello_sent and Time.get_ticks_usec() - connected_us > 10000000):
		_fail("Local MuJoCo response timed out. Simulation stopped.")


func _send(message: Dictionary) -> void:
	pending_request = message.duplicate(true)
	if peer.put_data((JSON.stringify(message) + "\n").to_utf8_buffer()) != OK:
		_fail("Local bridge write failed.")
		return
	pending = true
	last_request_us = Time.get_ticks_usec()


func _receive(message: Dictionary) -> void:
	var received_us := Time.get_ticks_usec()
	if message.get("type") == "error":
		_fail("MuJoCo: " + str(message.get("message", "request rejected")))
		return
	if pending and pending_kind == "ping" and message.get("type") == "pong" and message.get("model_sha256") == scene_data.bridge.model_sha256 and int(message.get("expected_sequence", -1)) == sequence:
		pending = false
		pending_kind = ""
		return
	if not pending or message.get("model_sha256") != scene_data.bridge.model_sha256:
		_fail("Unsolicited response or model identity mismatch.")
		return
	var was_reset := pending_kind == "reset"
	if pending_kind == "hello":
		if message.get("type") != "ready" or message.get("schema") != SCHEMA or float(message.get("physics_dt", -1)) != float(scene_data.bridge.physics_dt) or float(message.get("control_dt", -1)) != float(scene_data.bridge.control_dt):
			_fail("Local MuJoCo handshake differs from the scene.")
			return
		connected = true
		ready_us = received_us
		next_step_due_us = received_us
		for slider in sliders:
			slider.editable = true
	else:
		if message.get("type") != "state" or int(message.get("sequence", -1)) != sequence:
			_fail("Bridge response sequence mismatch.")
			return
		sequence += 1
		if was_reset:
			resets += 1
		else:
			replies += 1
		last_compute_ms = float(message.get("compute_ms", 0))
		compute_total_ms += last_compute_ms
		if rpc_roundtrip_ms.size() < 10000:
			rpc_roundtrip_ms.append((received_us - last_request_us) / 1000.0)
	var state: Variant = message.get("state")
	if not state is Dictionary or not state.get("body_positions") is Array or not state.get("body_quaternions") is Array or state.body_positions.size() != body_nodes.size() or state.body_quaternions.size() != body_nodes.size():
		_fail("Native body state framing mismatch.")
		return
	for index in range(body_nodes.size()):
		if not _valid_vector(state.body_positions[index], 3) or not _valid_quaternion(state.body_quaternions[index]):
			_fail("Native body pose is nonfinite.")
			return
		body_nodes[index].transform = Transform3D(native_basis(state.body_quaternions[index]), native_vector(state.body_positions[index]))
	var new_time := float(state.get("time", -1))
	if not is_finite(new_time) or new_time < 0:
		_fail("Native simulation clock is invalid.")
		return
	if pending_kind == "hello":
		initial_sim_time = new_time
		initial_state = state.duplicate(true)
	elif not was_reset and absf(new_time - state_time - float(scene_data.bridge.control_dt)) > 0.000001:
		_fail("Native simulation did not advance exactly one control interval.")
		return
	state_time = new_time
	if was_reset or (pending_kind == "step" and received_us > next_step_due_us):
		next_step_due_us = received_us + 20000
	if smoke_steps > 0 and pending_kind != "hello" and smoke_trace.size() < 256:
		smoke_trace.append({"request": pending_request.duplicate(true), "state": state.duplicate(true), "compute_ms": last_compute_ms, "rpc_roundtrip_ms": (received_us - last_request_us) / 1000.0})
	pending = false
	pending_kind = ""
	status_label.text = "Paused · MuJoCo connected" if paused else "Running · MuJoCo connected"
	time_label.text = "Simulation  %.2f s" % state_time
	compute_label.text = "Native step  %.2f ms · 20 ms simulated" % last_compute_ms
	if smoke_steps > 0 and replies >= smoke_steps:
		_finish_smoke()


func _slider_changed(value: float, index: int) -> void:
	if index < 3:
		target[index] = value
	else:
		jaw = value
	value_labels[index].text = "%.0f mm" % (value * 1000) if index < 3 else "%.0f%% open" % (value / 0.035 * 100)
	marker.position = native_vector(target)


func _set_slider(index: int, value: float) -> void:
	sliders[index].set_value_no_signal(value)
	_slider_changed(value, index)


func _toggle_pause() -> void:
	if not connected:
		return
	paused = not paused
	next_step_due_us = Time.get_ticks_usec() + 20000
	pause_button.text = "Resume" if paused else "Pause"
	status_label.text = "Pausing after current step…" if paused and pending else ("Paused · MuJoCo connected" if paused else "Running · MuJoCo connected")


func _home() -> void:
	if not connected:
		return
	for index in range(3):
		_set_slider(index, scene_data.bridge.home_target[index])
	_set_slider(3, scene_data.bridge.gripper_initial)


func _reset() -> void:
	if not connected:
		return
	reset_requested = true
	status_label.text = "Resetting after current step…"


func _smoke_target() -> void:
	# Bounded, deterministic movement and jaw changes verify more than a static
	# connection. The trace preserves the exact transmitted action for replay.
	var fraction := float(replies) / float(smoke_steps)
	var offset := [0.0, 0.0, 0.0]
	var opening := 0.035
	if fraction >= 0.333 and fraction < 0.666:
		offset = [0.02, 0.025, -0.025]
		opening = 0.025
	elif fraction >= 0.666:
		offset = [0.035, -0.015, 0.01]
		opening = 0.01
	for index in range(3):
		_set_slider(index, float(scene_data.bridge.home_target[index]) + float(offset[index]))
	_set_slider(3, opening)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT:
			orbiting = event.pressed
		elif event.pressed and event.button_index == MOUSE_BUTTON_WHEEL_UP:
			camera_distance = maxf(0.45, camera_distance * 0.9)
		elif event.pressed and event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			camera_distance = minf(4.5, camera_distance * 1.1)
		_position_camera()
	elif event is InputEventMouseMotion and orbiting:
		camera_yaw -= event.relative.x * 0.006
		camera_pitch = clampf(camera_pitch + event.relative.y * 0.006, 0.08, 1.35)
		_position_camera()


func _position_camera() -> void:
	if camera == null:
		return
	camera.position = focus + camera_distance * Vector3(cos(camera_pitch) * sin(camera_yaw), sin(camera_pitch), cos(camera_pitch) * cos(camera_yaw))
	camera.look_at(focus)
	# Shift the scene away from the overlaid controls without moving its geometry.
	camera.h_offset = 0.17


func _fail(message: String) -> void:
	if errors.is_empty() or errors[-1] != message:
		errors.append(message)
	paused = true
	if status_label != null:
		status_label.text = message
		status_label.add_theme_color_override("font_color", Color("a14d43"))
	for slider in sliders:
		slider.editable = false
	peer.disconnect_from_host()
	push_error(message)
	if smoke_steps > 0 or not smoke_output.is_empty():
		_finish_smoke()


func _finish_smoke() -> void:
	if finished:
		return
	finished = true
	completed_us = Time.get_ticks_usec()
	peer.disconnect_from_host()
	if not screenshot_path.is_empty() and DisplayServer.get_name() != "headless":
		await RenderingServer.frame_post_draw
		if FileAccess.file_exists(screenshot_path):
			errors.append("Existing screenshot was preserved.")
		else:
			var saved := get_viewport().get_texture().get_image().save_png(screenshot_path)
			if saved != OK:
				errors.append("Screenshot could not be written.")
	var summary := {
		"schema": "dvidia.godot-client-smoke/v1", "ok": errors.is_empty(),
		"client_script_sha256": FileAccess.get_sha256("res://main.gd"),
		"project_sha256": FileAccess.get_sha256("res://project.godot"),
		"processor": OS.get_processor_name(), "logical_processors": OS.get_processor_count(),
		"expected_model_sha256": scene_data.get("bridge", {}).get("model_sha256", ""),
		"scene_integrity_sha256": scene_integrity_sha256,
		"step_responses": replies, "reset_responses": resets, "requested_steps": smoke_steps,
		"initial_sim_time_seconds": initial_sim_time, "applied_sim_time_seconds": state_time,
		"initial_state": initial_state, "trace": smoke_trace, "trace_limit": 256,
		"wall_seconds": (Time.get_ticks_usec() - started_us) / 1000000.0,
		"interactive_wall_seconds": (completed_us - ready_us) / 1000000.0 if ready_us > 0 else 0.0,
		"process_frame_interval_ms": _distribution(frame_intervals_ms),
		"rpc_roundtrip_ms": _distribution(rpc_roundtrip_ms),
		"timing_scope": "Startup-inclusive wall time; ready-to-completion interactive wall; Godot process delta intervals (not GPU render time); socket request-to-parsed-response latency; native compute excludes IPC and rendering. Samples capped at 10000 per series.",
		"native_compute_total_ms": compute_total_ms, "render_process_frames": frame_count,
		"physics_authority": "MuJoCo only", "godot": Engine.get_version_info().string,
		"errors": errors,
	}
	if not smoke_output.is_empty():
		if FileAccess.file_exists(smoke_output):
			push_error("Existing smoke output was preserved.")
			get_tree().quit(1)
			return
		var output := FileAccess.open(smoke_output, FileAccess.WRITE)
		if output == null:
			push_error("Cannot write smoke output.")
			get_tree().quit(1)
			return
		output.store_string(JSON.stringify(summary, "  ") + "\n")
	var compact := summary.duplicate()
	compact.erase("initial_state")
	compact.erase("trace")
	print(JSON.stringify(compact))
	get_tree().quit(0 if errors.is_empty() else 1)


func _distribution(values: Array[float]) -> Dictionary:
	if values.is_empty():
		return {"samples": 0, "p50": null, "p95": null}
	var ordered := values.duplicate()
	ordered.sort()
	return {"samples": values.size(), "p50": ordered[int(floor((ordered.size() - 1) * 0.5))], "p95": ordered[int(ceil((ordered.size() - 1) * 0.95))]}
