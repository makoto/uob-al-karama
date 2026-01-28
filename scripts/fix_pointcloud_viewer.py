#!/usr/bin/env python3
"""Fix pointcloud viewer by embedding data directly in HTML."""

import random

# Read PLY file
ply_path = 'output/pointcloud/test_area_pointcloud.ply'
print(f"Reading {ply_path}...")

with open(ply_path, 'r') as f:
    lines = f.readlines()

# Parse header
header_end = 0
vertex_count = 0
for i, line in enumerate(lines):
    if line.startswith('element vertex'):
        vertex_count = int(line.split()[2])
    if line.strip() == 'end_header':
        header_end = i + 1
        break

print(f"  Total vertices: {vertex_count}")

# Parse vertices
vertices = []
for i in range(vertex_count):
    line = lines[header_end + i].strip()
    if not line:
        continue
    parts = line.split()
    if len(parts) >= 6:
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
        vertices.append((x, y, z, r, g, b))

# Sample if too many points
max_points = 50000
if len(vertices) > max_points:
    print(f"  Sampling {max_points} points...")
    vertices = random.sample(vertices, max_points)

print(f"  Using {len(vertices)} points")

# Convert to JavaScript arrays
positions = []
colors = []
for x, y, z, r, g, b in vertices:
    # Swap Y and Z for Three.js coordinate system
    positions.extend([round(x, 2), round(z, 2), round(-y, 2)])
    colors.extend([round(r/255, 3), round(g/255, 3), round(b/255, 3)])

html = f'''<!DOCTYPE html>
<html>
<head>
    <title>3D Point Cloud Viewer - Al Karama Test Area</title>
    <style>
        body {{ margin: 0; overflow: hidden; background: #1a1a1a; }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            font-family: Arial, sans-serif;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 8px;
            z-index: 100;
        }}
        #info h3 {{ margin: 0 0 10px 0; }}
        #info p {{ margin: 5px 0; font-size: 14px; }}
    </style>
</head>
<body>
    <div id="info">
        <h3>Point Cloud Viewer</h3>
        <p>Test Area: 26 SVI images</p>
        <p>Points: {len(vertices):,}</p>
        <p><b>Controls:</b></p>
        <p>Left drag: Rotate</p>
        <p>Right drag: Pan</p>
        <p>Scroll: Zoom</p>
        <p style="color: #f0ad4e; margin-top: 10px;">Note: Without compass data,<br>all images face same direction</p>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

    <script>
        // Embedded point data
        const positions = new Float32Array({positions});
        const colors = new Float32Array({colors});

        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a1a);

        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 50, 100);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        // Add axes helper
        const axesHelper = new THREE.AxesHelper(20);
        scene.add(axesHelper);

        // Add grid
        const gridHelper = new THREE.GridHelper(100, 20, 0x444444, 0x333333);
        scene.add(gridHelper);

        // Create point cloud
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({{
            size: 0.5,
            vertexColors: true,
            sizeAttenuation: true
        }});

        const points = new THREE.Points(geometry, material);
        scene.add(points);

        // Center camera on point cloud
        geometry.computeBoundingBox();
        const center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        controls.target.copy(center);

        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}

        // Handle resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        animate();
    </script>
</body>
</html>'''

output_path = 'output/pointcloud_viewer.html'
print(f"Writing {output_path}...")
with open(output_path, 'w') as f:
    f.write(html)

print("Done!")
