const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
    45,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
);

const renderer = new THREE.WebGLRenderer({
    canvas: document.querySelector("#bg"),
    antialias: true
});

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
camera.position.z = 8;

/* Deep background color */
scene.background = new THREE.Color(0x050816);

/* Arc */
const curve = new THREE.EllipseCurve(
    0, 0,
    4, 4,
    Math.PI * 0.15,
    Math.PI * 0.85,
    false,
    0
);

const points = curve.getPoints(200);
const geometry = new THREE.BufferGeometry().setFromPoints(points);

const material = new THREE.LineBasicMaterial({
    color: 0x4f8cff
});

const arc = new THREE.Line(geometry, material);
scene.add(arc);

/* Glow sphere */
const sphereGeometry = new THREE.SphereGeometry(0.6, 32, 32);
const sphereMaterial = new THREE.MeshBasicMaterial({
    color: 0x4f8cff
});
const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
sphere.position.set(0, -1.5, 0);
scene.add(sphere);

/* Bloom */
const composer = new THREE.EffectComposer(renderer);
composer.addPass(new THREE.RenderPass(scene, camera));

const bloomPass = new THREE.UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    1.5,
    0.4,
    0.85
);

composer.addPass(bloomPass);

/* Animation */
function animate() {
    requestAnimationFrame(animate);
    arc.rotation.z += 0.002;
    composer.render();
}

animate();

/* Resize */
window.addEventListener("resize", () => {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
});