import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Scene Setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x222222);
scene.fog = new THREE.Fog(0x222222, 20, 150);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
// Camera position: Behind and Up
camera.position.set(-10, 5, 10);
camera.up.set(0, 1, 0); // Y-Up

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// Lights
const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444);
scene.add(hemiLight);
const dirLight = new THREE.DirectionalLight(0xffffff);
dirLight.position.set(0, 20, 10);
scene.add(dirLight);

// Ground
const grid = new THREE.GridHelper(200, 200, 0x555555, 0x333333);
scene.add(grid);

// Drone Mesh
const droneGroup = new THREE.Group();
const bodyGeo = new THREE.BoxGeometry(0.5, 0.1, 0.5);
const bodyMat = new THREE.MeshPhongMaterial({ color: 0x00aaff });
const body = new THREE.Mesh(bodyGeo, bodyMat);
droneGroup.add(body);

const armGeo = new THREE.BoxGeometry(0.8, 0.05, 0.05);
const arm1 = new THREE.Mesh(armGeo, bodyMat);
arm1.rotation.y = Math.PI / 4;
droneGroup.add(arm1);
const arm2 = new THREE.Mesh(armGeo, bodyMat);
arm2.rotation.y = -Math.PI / 4;
droneGroup.add(arm2);

const axesHelper = new THREE.AxesHelper(1);
droneGroup.add(axesHelper);
scene.add(droneGroup);

// Target Mesh
const targetGeo = new THREE.SphereGeometry(0.5, 16, 16);
const targetMat = new THREE.MeshPhongMaterial({ color: 0xff0000 });
const target = new THREE.Mesh(targetGeo, targetMat);
scene.add(target);

// Ghosts
const ghostGroup = new THREE.Group();
scene.add(ghostGroup);

// Telemetry UI
const statusEl = document.getElementById('status');
const telemEl = document.getElementById('telemetry');

// WebSocket
const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
const ws = new WebSocket(`${protocol}://${window.location.host}/ws`);

ws.onopen = () => {
    statusEl.innerText = "Connected";
    statusEl.style.color = "#0f0";
};

ws.onclose = () => {
    statusEl.innerText = "Disconnected";
    statusEl.style.color = "#f00";
};

ws.onmessage = (event) => {
    try {
        const data = JSON.parse(event.data);
        updateState(data);
    } catch (e) {
        console.error("Parse Error", e);
    }
};

function updateState(data) {
    // Data: { drone: {px, py, pz, roll, pitch, yaw}, target: [x,y,z], ghosts: [...] }

    // Coordinate Conversion: NED (Sim) to Y-Up (ThreeJS)
    // Sim X (North) -> Three -Z
    // Sim Y (East)  -> Three X
    // Sim Z (Down)  -> Three -Y

    function toThree(p) {
        if (!p) return new THREE.Vector3();
        // Input: [px, py, pz] or {px, py, pz}
        let x, y, z;
        if (Array.isArray(p)) { x=p[0]; y=p[1]; z=p[2]; }
        else { x=p.px; y=p.py; z=p.pz; }
        return new THREE.Vector3(y, -z, -x);
    }

    // Drone
    const d = data.drone;
    const pos = toThree(d);
    droneGroup.position.copy(pos);

    // Rotation
    // Sim Euler: Roll, Pitch, Yaw.
    // Three Euler: X, Y, Z.
    // Sim Pitch (around East/Y) -> Three X rotation.
    // Sim Yaw (around Down/Z) -> Three Y rotation (Need to invert? Down is -Y. Yaw is usually CW in NED. Three Y is CCW around Up. So -Yaw.)
    // Sim Roll (around North/X) -> Three Z rotation (North is -Z. Roll is CW? Three Z is CCW. So -Roll?)

    droneGroup.rotation.set(d.pitch, -d.yaw, -d.roll);

    // Target
    if (data.target) {
        const tPos = toThree(data.target);
        target.position.copy(tPos);
    }

    // Ghosts
    ghostGroup.clear();
    if (data.ghosts) {
        data.ghosts.forEach((path, idx) => {
            const points = path.map(pt => toThree(pt));
            if (points.length > 0) {
                const geo = new THREE.BufferGeometry().setFromPoints(points);
                // Color ramp or different colors per model?
                // Let's make the "Best" (first?) model Green, others White/Gray.
                const color = idx === 0 ? 0x00ff00 : 0xaaaaaa;
                const mat = new THREE.LineBasicMaterial({ color: color, opacity: 0.5, transparent: true });
                const line = new THREE.Line(geo, mat);
                ghostGroup.add(line);
            }
        });
    }

    // Telemetry
    telemEl.innerText = `
        Alt: ${(-d.pz).toFixed(2)} m
        State: ${data.state || "N/A"}
        Vel: ${d.vx?.toFixed(1) || 0}
    `;
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
