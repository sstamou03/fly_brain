import * as THREE from "three";

function canvasTexture(w: number, h: number, draw: (g: CanvasRenderingContext2D) => void, color = true) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  draw(c.getContext("2d")!);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = color ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  t.anisotropy = 8;
  return t;
}

function rng(seed: number) {
  let s = seed;
  return () => ((s = (s * 16807) % 2147483647) - 1) / 2147483646;
}

/** Wood planks with grain, repeated. */
export function woodTexture(base: string, dark: string, planks: number, repeat: [number, number], seed = 7) {
  const r = rng(seed);
  const t = canvasTexture(1024, 1024, (g) => {
    const pw = 1024 / planks;
    for (let p = 0; p < planks; p++) {
      g.fillStyle = base;
      g.fillRect(p * pw, 0, pw, 1024);
      g.globalAlpha = 0.18 + r() * 0.2;
      g.fillStyle = r() > 0.5 ? dark : "#000";
      g.fillRect(p * pw, 0, pw, 1024);
      g.globalAlpha = 1;
      for (let k = 0; k < 38; k++) {
        const x = p * pw + r() * pw;
        g.strokeStyle = `rgba(0,0,0,${0.05 + r() * 0.14})`;
        g.lineWidth = 0.6 + r() * 2.2;
        g.beginPath();
        g.moveTo(x, 0);
        for (let y = 0; y <= 1024; y += 64) g.lineTo(x + Math.sin(y / (90 + r() * 60) + k) * (3 + r() * 6), y);
        g.stroke();
      }
      g.fillStyle = "rgba(0,0,0,0.55)";
      g.fillRect(p * pw, 0, 2.5, 1024);
    }
  });
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(...repeat);
  return t;
}

/** Dark wood wall panels with raised frames. */
export function panelTexture() {
  const t = canvasTexture(1024, 512, (g) => {
    g.fillStyle = "#24160e";
    g.fillRect(0, 0, 1024, 512);
    for (let i = 0; i < 4; i++) {
      const x = 24 + i * 250;
      g.fillStyle = "#2e1d13";
      g.fillRect(x, 40, 226, 430);
      g.strokeStyle = "rgba(0,0,0,0.6)";
      g.lineWidth = 6;
      g.strokeRect(x, 40, 226, 430);
      g.strokeStyle = "rgba(255,200,140,0.06)";
      g.lineWidth = 2;
      g.strokeRect(x + 10, 50, 206, 410);
    }
  });
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(3, 1);
  return t;
}

/** Framed poster for the wall. */
export function posterTexture(title: string, sub: string, ink: string, paper: string) {
  return canvasTexture(512, 700, (g) => {
    g.fillStyle = "#1a100a";
    g.fillRect(0, 0, 512, 700);
    g.fillStyle = paper;
    g.fillRect(28, 28, 456, 644);
    g.fillStyle = ink;
    g.textAlign = "center";
    g.font = "bold 74px Georgia, serif";
    title.split("\n").forEach((line, i) => g.fillText(line, 256, 220 + i * 84));
    g.font = "28px Georgia, serif";
    g.fillText(sub, 256, 560);
    g.strokeStyle = ink;
    g.lineWidth = 3;
    g.strokeRect(58, 58, 396, 584);
  });
}

/** Drosophila abdomen: tan with dark tergite bands, bands run around the body (along v). */
export function abdomenTexture() {
  return canvasTexture(64, 512, (g) => {
    g.fillStyle = "#b58a52";
    g.fillRect(0, 0, 64, 512);
    for (let i = 0; i < 6; i++) {
      const y = 70 + i * 62;
      const grad = g.createLinearGradient(0, y, 0, y + 40);
      grad.addColorStop(0, "rgba(35,22,12,0.0)");
      grad.addColorStop(0.35, "rgba(35,22,12,0.92)");
      grad.addColorStop(1, "rgba(35,22,12,0.15)");
      g.fillStyle = grad;
      g.fillRect(0, y, 64, 40);
    }
  });
}

/** Hexagonal ommatidia bump pattern for the compound eyes. */
export function facetTexture() {
  const t = canvasTexture(512, 512, (g) => {
    g.fillStyle = "#808080";
    g.fillRect(0, 0, 512, 512);
    const s = 9;
    for (let row = 0; row < 70; row++) {
      for (let col = 0; col < 70; col++) {
        const x = col * s * 1.5;
        const y = row * s * 1.732 + (col % 2 ? s * 0.866 : 0);
        const grad = g.createRadialGradient(x, y, 0, x, y, s);
        grad.addColorStop(0, "#ffffff");
        grad.addColorStop(0.8, "#6a6a6a");
        grad.addColorStop(1, "#303030");
        g.fillStyle = grad;
        g.beginPath();
        for (let k = 0; k < 6; k++) {
          const a = (Math.PI / 3) * k;
          g.lineTo(x + Math.cos(a) * s, y + Math.sin(a) * s);
        }
        g.fill();
      }
    }
  }, false);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(2, 1);
  return t;
}

/** Wing membrane with Drosophila-like longitudinal veins (L1-L5) and two cross-veins. */
export function wingTexture() {
  return canvasTexture(256, 512, (g) => {
    g.clearRect(0, 0, 256, 512);
    g.fillStyle = "rgba(210,225,235,0.35)";
    g.fillRect(0, 0, 256, 512);
    g.strokeStyle = "rgba(60,45,30,0.85)";
    g.lineCap = "round";
    const veins: [number, number, number, number][] = [
      [128, 0, 20, 470], [128, 0, 70, 505], [128, 0, 125, 512], [128, 0, 185, 495], [128, 0, 235, 420],
    ];
    veins.forEach(([x0, y0, x1, y1], i) => {
      g.lineWidth = i === 0 ? 5 : 3;
      g.beginPath();
      g.moveTo(x0, y0);
      g.quadraticCurveTo((x0 + x1) / 2 + (i - 2) * 18, y1 * 0.45, x1, y1);
      g.stroke();
    });
    g.lineWidth = 2.5;
    g.beginPath();
    g.moveTo(96, 250);
    g.lineTo(140, 262);
    g.moveTo(150, 330);
    g.lineTo(196, 322);
    g.stroke();
  });
}
