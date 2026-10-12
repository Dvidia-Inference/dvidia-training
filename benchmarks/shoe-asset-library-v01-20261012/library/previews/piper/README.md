# DVIDIA asset library preview

Recorded native MuJoCo scene settling with held arm commands. The browser displays saved poses and does not run physics. Shoes are authored rigid proxies with untested physical properties. No learned shoe skill, calibrated deformation, hardware or real-time qualification.

Serve this directory with `python -m http.server 8000` and open http://localhost:8000. Everything is local; JavaScript, WebGL2 and DecompressionStream are required. The pair selector filters recorded objects visually; use a downloaded scene descriptor with dvidia-assets scene --config to compile and simulate only that pair. geometry.bin.gz and motion.bin.gz decode to bounded little-endian Float32 positions and wxyz quaternions. MIT applies to DVIDIA code and synthetic shoes. Robot model licenses and original READMEs remain under licenses/. Three.js/OrbitControls notices are included.
