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

// Actual Target Mesh (Green)
const simTargetMat = new THREE.MeshPhongMaterial({ color: 0x00ff00, transparent: true, opacity: 0.5 });
const simTarget = new THREE.Mesh(targetGeo, simTargetMat);
scene.add(simTarget);

// Ghosts
const ghostGroup = new THREE.Group();
scene.add(ghostGroup);

// Telemetry UI
const statusEl = document.getElementById('status');
const debugTelemEl = document.getElementById('debug-telem');
const debugControlEl = document.getElementById('debug-control');
const debugMissionEl = document.getElementById('debug-mission');
const hudCanvas = document.getElementById('hud-canvas');
const ctx = hudCanvas.getContext('2d');

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
        console.log("WebSocket Received:", {
            drone: data.drone,
            target: data.target,
            control: data.control
        });
        updateState(data);
    } catch (e) {
        console.error("Parse Error", e);
    }
};

// UI Controls
const resetBtn = document.getElementById('reset-btn');
const initAltInput = document.getElementById('init-alt');
const initDistInput = document.getElementById('init-dist');

const pauseBtn = document.getElementById('pause-btn');
const resumeBtn = document.getElementById('resume-btn');
const stepBtn = document.getElementById('step-btn');

const speedSlider = document.getElementById('speed-slider');
const speedVal = document.getElementById('speed-val');

const tgtXInput = document.getElementById('tgt-x');
const tgtYInput = document.getElementById('tgt-y');
const tgtZInput = document.getElementById('tgt-z');
const updateTgtBtn = document.getElementById('update-tgt-btn');

function sendMsg(type, payload = {}) {
    if (ws.readyState === WebSocket.OPEN) {
        const msg = JSON.stringify({ type, ...payload });
        ws.send(msg);
        console.log("Sent:", msg);
    } else {
        console.error("WebSocket not connected");
    }
}

resetBtn.addEventListener('click', () => {
    const alt = parseFloat(initAltInput.value);
    const dist = parseFloat(initDistInput.value);
    sendMsg('reset', { altitude: alt, distance: dist });
});

pauseBtn.addEventListener('click', () => sendMsg('pause'));
resumeBtn.addEventListener('click', () => sendMsg('resume'));
stepBtn.addEventListener('click', () => sendMsg('step'));

speedSlider.addEventListener('input', () => {
    const val = parseFloat(speedSlider.value);
    speedVal.innerText = val.toFixed(1);
    sendMsg('set_speed', { speed: val });
});

updateTgtBtn.addEventListener('click', () => {
    const x = parseFloat(tgtXInput.value);
    const y = parseFloat(tgtYInput.value);
    const z = parseFloat(tgtZInput.value);
    sendMsg('update_target', { x, y, z });
});

function updateState(data) {
    // Check Paused State
    if (data.paused !== undefined) {
        if (data.paused) {
            statusEl.innerText = "Connected (PAUSED)";
            statusEl.style.color = "yellow";
            pauseBtn.disabled = true;
            resumeBtn.disabled = false;
            stepBtn.disabled = false;
        } else {
            statusEl.innerText = "Connected (RUNNING)";
            statusEl.style.color = "#0f0";
            pauseBtn.disabled = false;
            resumeBtn.disabled = true;
            stepBtn.disabled = true;
        }
    }
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

    // Sim Target
    if (data.sim_target) {
        const stPos = toThree(data.sim_target);
        simTarget.position.copy(stPos);

        // Update Inputs if not focused? Or just leave them?
        // Let's not overwrite user input while typing.
        if (document.activeElement !== tgtXInput && document.activeElement !== tgtYInput && document.activeElement !== tgtZInput) {
            tgtXInput.value = data.sim_target[0];
            tgtYInput.value = data.sim_target[1];
            tgtZInput.value = data.sim_target[2];
        }
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

    // Telemetry & Debug
    function row(label, val) {
        return `<div class="debug-row"><span>${label}:</span> <span class="debug-val">${val}</span></div>`;
    }

    // Flight Data
    const alt = -d.pz;
    const vel = Math.sqrt((d.vx||0)**2 + (d.vy||0)**2 + (d.vz||0)**2);

    debugTelemEl.innerHTML =
        row('Alt', alt.toFixed(2) + ' m') +
        row('Speed', vel.toFixed(2) + ' m/s') +
        row('Vx', (d.vx||0).toFixed(2)) +
        row('Vy', (d.vy||0).toFixed(2)) +
        row('Vz', (d.vz||0).toFixed(2)) +
        row('Roll', (d.roll * 180/Math.PI).toFixed(1) + ' deg') +
        row('Pitch', (d.pitch * 180/Math.PI).toFixed(1) + ' deg') +
        row('Yaw', (d.yaw * 180/Math.PI).toFixed(1) + ' deg');

    // Control
    if (data.control) {
        const c = data.control;
        debugControlEl.innerHTML =
            row('Thrust', c.thrust?.toFixed(3)) +
            row('Roll R', c.roll_rate?.toFixed(3)) +
            row('Pitch R', c.pitch_rate?.toFixed(3)) +
            row('Yaw R', c.yaw_rate?.toFixed(3));
    }

    // Mission / Error
    let errStr = "";
    if (data.dpc_error && data.dpc_error.height_error !== undefined) {
         const e = data.dpc_error;
         errStr = `<div style="margin-top:5px; border-top:1px dashed #555;">
         <div style="color:#faa">Pred Error (T=${e.pred_time}s)</div>
         ${row('H Err', e.height_error)}
         ${row('Size Err', e.size_error)}
         ${row('U Err', e.u_error)}
         ${row('V Err', e.v_error)}
         </div>`;
    }

    debugMissionEl.innerHTML =
        row('State', data.state || "N/A") +
        (data.paused ? row('Status', 'PAUSED') : "") +
        errStr;

    // HUD
    // Clear
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, 240, 180);

    // Draw Center Cross
    ctx.strokeStyle = '#333';
    ctx.beginPath();
    ctx.moveTo(120, 0); ctx.lineTo(120, 180);
    ctx.moveTo(0, 90); ctx.lineTo(240, 90);
    ctx.stroke();

    // Draw Actual Tracker Blob (Red)
    if (data.tracker && data.tracker.size > 0) {
        const u = data.tracker.u;
        const v = data.tracker.v;
        const r = data.tracker.size;

        // Scale 640x480 to 240x180 (Scale 0.375)
        const scale = 240 / 640;

        ctx.beginPath();
        ctx.arc(u * scale, v * scale, r * scale, 0, 2 * Math.PI);
        ctx.fillStyle = 'red';
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.stroke();
    }
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
