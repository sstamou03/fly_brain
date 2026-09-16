import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { loadActivity, loadBrain, sampleSeries, type BarData, type BrainPoints } from "../data";
import { tasteProgress, tour, useTour } from "../store";

const vertex = /* glsl */ `
  attribute float group;
  attribute float actA;
  attribute float actB;
  uniform float uMix;
  uniform float uPulse;
  uniform float uTime;
  uniform float uSize;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float a = mix(actA, actB, uMix) / 255.0;
    float flicker = 0.6 + 0.4 * sin(uTime * 11.0 + position.x * 53.0 + position.z * 37.0);
    float glow = clamp(a * (0.6 + 0.8 * uPulse) * flicker * 2.2, 0.0, 1.0);
    vec3 base = group < 0.5 ? vec3(0.21, 0.25, 0.32)
              : group < 1.5 ? vec3(0.15, 0.21, 0.29)
              : group < 2.5 ? vec3(0.19, 0.20, 0.28)
              : group < 3.5 ? vec3(0.28, 0.24, 0.21)
              : vec3(0.25, 0.21, 0.28);
    vec3 hot = mix(vec3(1.0, 0.50, 0.14), vec3(1.0, 0.95, 0.82), glow * glow);
    vColor = mix(base, hot, glow);
    vAlpha = 0.14 + 0.86 * glow;
    gl_PointSize = uSize * (1.0 + 3.4 * glow) / -mv.z;
    gl_Position = projectionMatrix * mv;
  }
`;

const fragment = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float d = length(c);
    if (d > 0.5) discard;
    gl_FragColor = vec4(vColor, vAlpha * (1.0 - smoothstep(0.1, 0.5, d)));
  }
`;

function Cloud({ brain, data }: { brain: BrainPoints; data: BarData }) {
  const group = useRef<THREE.Group>(null!);
  const drinkIndex = useTour((s) => s.current ?? s.target ?? -1);
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(brain.xyz, 3));
    g.setAttribute("group", new THREE.BufferAttribute(Float32Array.from(brain.group), 1));
    g.setAttribute("actA", new THREE.BufferAttribute(new Float32Array(brain.count), 1));
    g.setAttribute("actB", new THREE.BufferAttribute(new Float32Array(brain.count), 1));
    return g;
  }, [brain]);
  const material = useMemo(
    () => new THREE.ShaderMaterial({
      vertexShader: vertex, fragmentShader: fragment, transparent: true, depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: { uMix: { value: 1 }, uPulse: { value: 0 }, uTime: { value: 0 }, uSize: { value: 9 * Math.min(window.devicePixelRatio, 2) } },
    }),
    [],
  );

  // cross-fade to the activity of the drink the fly is at
  useEffect(() => {
    const drink = data.drinks[drinkIndex];
    const file = drink?.cloud?.file;
    let cancelled = false;
    const apply = (next: Float32Array) => {
      if (cancelled) return;
      const a = geometry.getAttribute("actA") as THREE.BufferAttribute;
      const b = geometry.getAttribute("actB") as THREE.BufferAttribute;
      const m = material.uniforms.uMix.value as number;
      for (let i = 0; i < brain.count; i++) (a.array as Float32Array)[i] = (a.array as Float32Array)[i] * (1 - m) + (b.array as Float32Array)[i] * m;
      (b.array as Float32Array).set(next);
      a.needsUpdate = true;
      b.needsUpdate = true;
      material.uniforms.uMix.value = 0;
    };
    if (!file) apply(new Float32Array(brain.count));
    else loadActivity(file).then((u8) => apply(Float32Array.from(u8)));
    return () => {
      cancelled = true;
    };
  }, [drinkIndex, data, brain, geometry, material]);

  useFrame((state, dt) => {
    const s = tour.get();
    const u = material.uniforms;
    u.uTime.value = state.clock.elapsedTime;
    u.uMix.value = Math.min(1, (u.uMix.value as number) + dt * 1.6);
    const drink = s.current !== null ? data.drinks[s.current] : null;
    const tasting = s.phase === "tasting" || s.phase === "settled";
    const target = drink && tasting ? 0.35 + 0.65 * sampleSeries(drink.pulse, tasteProgress(performance.now())) : drink ? 0.25 : 0;
    u.uPulse.value += (target - (u.uPulse.value as number)) * (1 - Math.exp(-dt * 6));
    group.current.rotation.y += dt * 0.12;
  });

  return (
    <group ref={group} rotation={[Math.PI, 0, 0]}>
      <points geometry={geometry} material={material} />
    </group>
  );
}

export function BrainView({ data }: { data: BarData }) {
  const [brain, setBrain] = useState<BrainPoints | null>(null);
  useEffect(() => {
    loadBrain(data.points).then(setBrain).catch(() => setBrain(null));
  }, [data.points]);
  return (
    <div className="brain-canvas">
      {brain ? (
        <Canvas dpr={[1, 2]} camera={{ position: [0, 0.2, 2.4], fov: 42 }}>
          <Cloud brain={brain} data={data} />
          <OrbitControls enableZoom={false} enablePan={false} rotateSpeed={0.5} />
        </Canvas>
      ) : (
        <p className="brain-loading">Φορτώνει ο εγκέφαλος…</p>
      )}
    </div>
  );
}
