import * as THREE from "three";
import { COUNTER_TOP } from "../layout";

/** Where the fly is in the world, written by the fly every frame and read by the camera. */
export const flyWorld = new THREE.Vector3(-2.9, COUNTER_TOP, 0.55);
