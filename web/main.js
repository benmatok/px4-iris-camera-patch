import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x222222);
scene.fog = new THREE.Fog(0x222222, 20, 200);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 5, 10);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// Ground Grid
const gridHelper = new THREE.GridHelper(200, 200, 0x444444, 0x333333);
scene.add(gridHelper);

const axesHelper = new THREE.AxesHelper(2);
scene.add(axesHelper);

// Drone Model
const droneGroup = new THREE.Group();
const droneBody = new THREE.Mesh(
    new THREE.BoxGeometry(0.6, 0.1, 0.6),
    new THREE.MeshLambertMaterial({ color: 0x00aaff })
);
droneGroup.add(droneBody);

// Direction Indicator (Front Arms)
const droneFront = new THREE.Mesh(
    new THREE.BoxGeometry(0.1, 0.1, 0.2),
    new THREE.MeshBasicMaterial({ color: 0xffaa00 })
);
droneFront.position.set(0, 0, -0.3); // -Z is Forward in Three.js Local
droneGroup.add(droneFront);
scene.add(droneGroup);

// Target Model
const targetMesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.5, 16, 16),
    new THREE.MeshBasicMaterial({ color: 0xff0000 })
);
scene.add(targetMesh);

// Ghosts Container
const ghostLines = new THREE.Group();
scene.add(ghostLines);

// Lights
const ambientLight = new THREE.AmbientLight(0x404040);
scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 1);
dirLight.position.set(10, 20, 10);
scene.add(dirLight);

// WebSocket
const infoDiv = document.getElementById('info');
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsHost = window.location.host || 'localhost:8080';
const wsUrl = `${wsProtocol}//${wsHost}/ws`;

const ws = new WebSocket(wsUrl);

ws.onopen = () => {
    infoDiv.innerText = `Connected to ${wsUrl}`;
};

ws.onclose = () => {
    infoDiv.innerText = "Disconnected";
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateState(data);
};

// Coordinate Mapping: Sim (NED) -> Three (Y-Up, -Z Forward)
// Sim X (North) -> Three -Z
// Sim Y (East)  -> Three X
// Sim Z (Down)  -> Three -Y
function mapCoord(x, y, z) {
    return new THREE.Vector3(y, -z, -x);
}

function updateState(data) {
    if (!data.drone) return;

    // Drone
    const p = data.drone.pos;
    const pos = mapCoord(p[0], p[1], p[2]);
    droneGroup.position.copy(pos);

    // Follow Camera (Optional)
    // camera.lookAt(pos);

    // Attitude (Simplified)
    // Sim Roll, Pitch, Yaw.
    // Three Rotation Order?
    // Let's just visualize Position primarily.

    // Target
    const t = data.target;
    const tPos = mapCoord(t[0], t[1], t[2]);
    targetMesh.position.copy(tPos);

    // Ghosts
    // Clear old lines
    while(ghostLines.children.length > 0){
        ghostLines.remove(ghostLines.children[0]);
    }

    if (data.ghosts) {
        data.ghosts.forEach((traj, idx) => {
            const points = traj.map(pt => mapCoord(pt[0], pt[1], pt[2]));
            const geometry = new THREE.BufferGeometry().setFromPoints(points);

            // Color code based on hypothesis?
            // 0: Nominal (Green), 1: Headwind (Blue), 2: Crosswind (Yellow)
            let color = 0x00ff00;
            if (idx === 1) color = 0x0000ff;
            if (idx === 2) color = 0xffff00;

            const material = new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.6 });
            const line = new THREE.Line(geometry, material);
            ghostLines.add(line);
        });
    }

    const alt = -p[2];
    infoDiv.innerHTML = `Connected<br>Alt: ${alt.toFixed(1)}m<br>Target Dist: ${pos.distanceTo(tPos).toFixed(1)}m`;
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
