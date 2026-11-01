"""Prompt-to-website starter generator for PyMind-A.

This is a deterministic, template-assisted compiler. It does not pretend that
the current tiny language model can author production websites autonomously.
"""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import re
from typing import Dict


@dataclass(frozen=True)
class WebsiteProject:
    name: str
    kind: str
    files: Dict[str, str]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:48] or "pymind-site"


def _kind(prompt: str) -> str:
    value = prompt.lower()
    if any(word in value for word in ("3d", "three-dimensional", "three dimensional", "car", "model viewer")):
        return "3d"
    if any(word in value for word in ("shop", "store", "product", "ecommerce", "e-commerce")):
        return "product"
    if any(word in value for word in ("portfolio", "resume", "personal", "developer")):
        return "portfolio"
    return "business"


def generate_website(prompt: str, project_name: str | None = None) -> WebsiteProject:
    """Generate a safe, editable starter site from a natural-language brief."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Please enter a website description.")
    if len(prompt) > 1200:
        raise ValueError("Website description must be 1,200 characters or fewer.")

    name = (project_name or prompt.strip().splitlines()[0][:48]).strip()
    name = name or "PyMind Website"
    safe_name = escape(name)
    safe_prompt = escape(prompt.strip())
    kind = _kind(prompt)

    if kind == "portfolio":
        heading = "Build a portfolio that tells your story."
        subheading = "Projects, skills and experience — presented clearly."
        cards = [("Selected work", "Showcase your strongest projects."),
                 ("Skills", "Highlight the tools and technologies you use."),
                 ("Contact", "Make it easy for people to reach you.")]
        action = "Explore projects"
    elif kind == "product":
        heading = "A better way to discover your next favourite."
        subheading = "A clean product landing page ready for your catalogue."
        cards = [("Featured product", "Add product details, images and pricing."),
                 ("Why choose us", "Explain the value your product provides."),
                 ("Get in touch", "Connect customers with your team.")]
        action = "View collection"
    elif kind == "3d":
        heading = "Explore a 3D scene in your browser."
        subheading = "Rotate the sample model, zoom in and inspect the scene."
        cards = [("Interactive preview", "Drag to rotate and scroll to zoom."),
                 ("Custom assets", "Replace the sample geometry with your GLB model."),
                 ("Next step", "Connect Blender exports from your PyMind workflow.")]
        action = "Explore the scene"
    else:
        heading = "A clear, modern home for your business."
        subheading = "Introduce what you do and help visitors take the next step."
        cards = [("What we do", "Describe your services in a few simple sentences."),
                 ("Why it matters", "Explain what makes your work valuable."),
                 ("Contact", "Give visitors a clear way to reach you.")]
        action = "Get started"

    card_html = "\n".join(
        f'<article class="card"><span class="eyebrow">0{index}</span><h3>{escape(title)}</h3><p>{escape(body)}</p></article>'
        for index, (title, body) in enumerate(cards, 1)
    )
    scene = '<div id="scene" class="scene" aria-label="Interactive 3D preview"><span>Loading 3D scene…</span></div>' if kind == "3d" else ""
    three_script = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>' if kind == "3d" else ""
    script = """document.querySelectorAll('[data-scroll]').forEach((button) => {
  button.addEventListener('click', () => document.querySelector('#details').scrollIntoView({behavior: 'smooth'}));
});
document.querySelector('#year').textContent = new Date().getFullYear();
"""
    if kind == "3d":
        script += """
const mount = document.querySelector('#scene');
if (window.THREE && mount) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#101827');
  const camera = new THREE.PerspectiveCamera(45, mount.clientWidth / mount.clientHeight, 0.1, 100);
  camera.position.set(4, 3, 6);
  const renderer = new THREE.WebGLRenderer({antialias: true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(mount.clientWidth, mount.clientHeight);
  mount.replaceChildren(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x26334a, 2));
  const key = new THREE.DirectionalLight(0xffffff, 2);
  key.position.set(4, 6, 5); scene.add(key);
  const body = new THREE.Mesh(new THREE.BoxGeometry(2.5, .65, 1.15), new THREE.MeshStandardMaterial({color: 0x2d9cdb, metalness: .25, roughness: .35}));
  body.position.y = .65; scene.add(body);
  const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.25, .65, .95), new THREE.MeshStandardMaterial({color: 0x9bd4ee, metalness: .1, roughness: .2}));
  cabin.position.set(-.15, 1.22, 0); scene.add(cabin);
  for (const x of [-.8, .8]) for (const z of [-.62, .62]) {
    const wheel = new THREE.Mesh(new THREE.CylinderGeometry(.32, .32, .18, 24), new THREE.MeshStandardMaterial({color: 0x20242b}));
    wheel.rotation.x = Math.PI / 2; wheel.position.set(x, .35, z); scene.add(wheel);
  }
  let dragging = false, previousX = 0, previousY = 0, angle = 0, tilt = 0;
  renderer.domElement.addEventListener('pointerdown', e => { dragging = true; previousX = e.clientX; previousY = e.clientY; });
  window.addEventListener('pointerup', () => dragging = false);
  window.addEventListener('pointermove', e => { if (!dragging) return; angle += (e.clientX - previousX) * .01; tilt = Math.max(-.5, Math.min(.5, tilt + (e.clientY - previousY) * .006)); previousX = e.clientX; previousY = e.clientY; });
  renderer.domElement.addEventListener('wheel', e => { camera.position.multiplyScalar(e.deltaY > 0 ? 1.08 : .92); e.preventDefault(); }, {passive:false});
  function animate() { requestAnimationFrame(animate); body.rotation.y = angle; cabin.rotation.y = angle; scene.rotation.x = tilt; renderer.render(scene, camera); }
  animate();
  new ResizeObserver(() => { camera.aspect = mount.clientWidth / mount.clientHeight; camera.updateProjectionMatrix(); renderer.setSize(mount.clientWidth, mount.clientHeight); }).observe(mount);
} else if (mount) mount.textContent = '3D preview needs an internet connection to load Three.js.';
"""
    index = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{safe_prompt[:150]}"><title>{safe_name} | PyMind-A</title>
<link rel="stylesheet" href="styles.css">{three_script}</head>
<body><header class="topbar"><a class="brand" href="#top">{safe_name}</a><button class="menu" data-scroll>{action}</button></header>
<main id="top"><section class="hero"><p class="eyebrow">PYMIND-A / GENERATED STARTER</p><h1>{heading}</h1><p class="lead">{subheading}</p><p class="brief">{safe_prompt}</p><button class="primary" data-scroll>{action} ↗</button></section>
<section id="details" class="details"><div class="section-heading"><p class="eyebrow">THE FOUNDATION</p><h2>Made to grow with your idea.</h2></div>{scene}<div class="grid">{card_html}</div></section></main>
<footer>Built with PyMind-A · <span id="year"></span></footer><script src="script.js"></script></body></html>"""
    css = """*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f5f7fb;color:#142033;font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}.topbar{height:76px;padding:0 clamp(20px,7vw,100px);display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #e2e8f0;background:#fff}.brand{font-weight:800;color:#142033;text-decoration:none;letter-spacing:-.04em}.menu,.primary{border:0;border-radius:999px;padding:13px 20px;font-weight:700;cursor:pointer}.menu{background:#edf2f8;color:#142033}.primary{background:#2457e6;color:#fff;margin-top:14px}.hero{padding:clamp(76px,12vw,150px) clamp(22px,12vw,170px);background:radial-gradient(circle at 85% 20%,#dce7ff,transparent 36%),#fff}.eyebrow{font-size:.72rem;letter-spacing:.16em;font-weight:800;color:#58709a}.hero h1{max-width:850px;font-size:clamp(2.8rem,7vw,6.2rem);line-height:.99;letter-spacing:-.065em;margin:20px 0}.lead{font-size:clamp(1.1rem,2vw,1.4rem);color:#53627a;max-width:650px}.brief{max-width:700px;color:#718096;line-height:1.7}.details{padding:90px clamp(22px,9vw,130px)}.section-heading h2{font-size:clamp(2rem,4vw,3.4rem);letter-spacing:-.05em;margin-top:8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px;margin-top:30px}.card{background:#fff;border:1px solid #e5eaf2;border-radius:22px;padding:28px;min-height:190px;box-shadow:0 12px 35px #15294a0a}.card h3{font-size:1.3rem}.card p{color:#64748b;line-height:1.6}.scene{height:380px;max-width:900px;margin:28px 0;border-radius:22px;overflow:hidden;background:#101827;color:#fff;display:grid;place-items:center}.scene canvas{display:block}footer{padding:28px;text-align:center;color:#718096;border-top:1px solid #e2e8f0}@media(max-width:600px){.topbar{height:64px}.menu{padding:10px 13px}.scene{height:280px}}"""
    return WebsiteProject(name=name, kind=kind, files={
        "index.html": index,
        "styles.css": css,
        "script.js": script,
        "README.txt": f"PyMind-A generated starter website\nType: {kind}\n\nOpen index.html in a browser.\nEdit styles.css and script.js to customize.\nPrompt: {prompt.strip()}\n",
    })


def write_website(project: WebsiteProject, output_dir: str | Path) -> Path:
    """Write generated files under the requested output directory."""
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    for relative, content in project.files.items():
        destination = (target / relative).resolve()
        if target not in destination.parents:
            raise ValueError("Invalid generated file path.")
        destination.write_text(content, encoding="utf-8")
    return target
