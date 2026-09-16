import { Suspense, useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { ContactShadows, Environment, Lightformer, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { BarData, WalkData } from "../data";
import { COUNTER_TOP, GLASS_Z, glassX } from "../layout";
import { tour, useTour } from "../store";
import { BarRoom } from "./BarRoom";
import { Fly } from "./Fly";
import { Glass } from "./Glass";
import { flyWorld } from "./shared";

/** Follows the fly: close on the glass it tastes, a little behind it while it wanders. */
function CameraRig({ n }: { n: number }) {
  const { camera, size } = useThree();
  const look = useRef(new THREE.Vector3(0, 1.4, 0));
  const pos = useMemo(() => new THREE.Vector3(), []);
  const tgt = useMemo(() => new THREE.Vector3(), []);
  useFrame((_, dt) => {
    const s = tour.get();
    const narrow = size.width < 700;
    const i = s.target ?? s.current;
    if (i === null) {
      pos.set(flyWorld.x * 0.85 + 0.2, COUNTER_TOP + 0.5, flyWorld.z + (narrow ? 2.2 : 1.5));
      tgt.set(flyWorld.x, COUNTER_TOP + 0.08, flyWorld.z);
    } else {
      const x = glassX(i, n);
      const far = s.phase === "flying" ? 0.5 : 0;
      const settled = s.phase === "settled" ? 0.35 : 0;
      pos.set(x * 0.8 + 0.18, COUNTER_TOP + 0.36 + far * 0.4 + settled * 0.3, GLASS_Z + (narrow ? 2.0 : 1.25) + far + settled);
      tgt.set(x, COUNTER_TOP + 0.16, GLASS_Z);
    }
    const k = 1 - Math.exp(-dt * 2.0);
    camera.position.lerp(pos, k);
    look.current.lerp(tgt, k);
    camera.lookAt(look.current);
  });
  return null;
}

export function BarScene({ data, walk }: { data: BarData; walk: WalkData | null }) {
  const spotTarget = useMemo(() => new THREE.Object3D(), []);
  const view = useTour((s) => s.view);
  const n = data.drinks.length;
  return (
    <Canvas
      dpr={[1, 2]}
      camera={{ position: [0, 1.85, 4.6], fov: 38, near: 0.01, far: 40 }}
      gl={{ antialias: true }}
      onCreated={({ gl }) => {
        gl.toneMapping = THREE.ACESFilmicToneMapping;
        gl.toneMappingExposure = 1.1;
      }}
    >
      <color attach="background" args={["#0c0806"]} />
      <fog attach="fog" args={["#0c0806", 7, 16]} />
      <ambientLight intensity={0.35} color="#ffdcb0" />
      <hemisphereLight args={["#ffd9a8", "#140b06", 0.45]} />
      <primitive object={spotTarget} position={[0, COUNTER_TOP, GLASS_Z]} />
      <spotLight target={spotTarget} position={[0.6, 4.1, 2.6]} angle={0.55} penumbra={0.9} intensity={60} decay={1.6} color="#ffd6a0" />
      <directionalLight position={[-3, 2.5, 5]} intensity={0.55} color="#c6d8ff" />
      <Suspense fallback={null}>
        <Environment resolution={256} frames={1}>
          <Lightformer form="rect" intensity={3} color="#ffb870" position={[0, 2.2, -1.5]} scale={[5, 1.8, 1]} />
          <Lightformer form="rect" intensity={1.5} color="#ffe2b8" position={[-4, 2.5, 2]} rotation={[0, Math.PI / 2, 0]} scale={[3, 2, 1]} />
          <Lightformer form="ring" intensity={4} color="#ffffff" position={[2.5, 3.5, 3]} scale={0.7} />
        </Environment>
        <BarRoom />
        {data.drinks.map((d, i) => (
          <Glass key={d.id} drink={d} index={i} n={n} provisional={data.provisional} />
        ))}
        <Fly data={data} walk={walk} />
        <ContactShadows position={[0, COUNTER_TOP + 0.002, GLASS_Z]} scale={[6.6, 0.9]} opacity={0.55} blur={2.2} far={0.6} resolution={512} />
      </Suspense>
      {view === "follow" ? (
        <CameraRig n={n} />
      ) : (
        <OrbitControls
          makeDefault
          target={[0, COUNTER_TOP + 0.35, 0]}
          minDistance={0.3}
          maxDistance={10}
          maxPolarAngle={Math.PI * 0.52}
          enableDamping
        />
      )}
    </Canvas>
  );
}
