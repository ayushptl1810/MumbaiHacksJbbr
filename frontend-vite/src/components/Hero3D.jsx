import { Suspense, useRef, useEffect } from "react";
import { Rive } from "rive-react";
import { Canvas, useFrame } from "@react-three/fiber";
import { useGLTF, Environment } from "@react-three/drei";
import * as THREE from "three";

// GLTF Model Component with continuous rotation
function GLTFModel({ modelPath }) {
  const { scene } = useGLTF(modelPath);
  const groupRef = useRef();

  // Continuous rotation - slower
  useFrame((state, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.2; // Slower rotation
    }
  });

  // Center and scale the model
  useEffect(() => {
    if (scene) {
      scene.traverse((child) => {
        if (child.isMesh) {
          // Disable shadows for better performance
          child.castShadow = false;
          child.receiveShadow = false;

          // Optimize material rendering
          if (child.material) {
            // Use simpler material properties if possible
            if (Array.isArray(child.material)) {
              child.material.forEach((mat) => {
                if (mat) {
                  mat.needsUpdate = false;
                }
              });
            } else if (child.material) {
              child.material.needsUpdate = false;
            }
          }

          // Enable frustum culling (already default, but explicit)
          child.frustumCulled = true;
        }
      });

      const box = new THREE.Box3().setFromObject(scene);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const scale = maxDim > 0 ? 2.3 / maxDim : 1; // Larger scale

      scene.scale.multiplyScalar(scale);
      scene.position.sub(center.multiplyScalar(scale));
    }
  }, [scene]);

  if (!scene) return null;

  return (
    <group ref={groupRef}>
      <primitive object={scene} />
    </group>
  );
}

// GLTF Viewer Component
export function GLTFViewer({ modelPath }) {
  if (!modelPath) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-black/20 rounded-lg border border-blue-500/20">
        <div className="text-center p-8">
          <div className="w-16 h-16 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400 mb-2 text-lg">3D Model Placeholder</p>
          <p className="text-xs text-gray-500 max-w-md">
            No model path provided. Add your GLTF file to src/assets/ and import
            it.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full" style={{ pointerEvents: "none" }}>
      <Canvas
        camera={{ position: [0, 0, 5], fov: 50 }}
        gl={{
          alpha: true,
          antialias: false, // Disable expensive antialiasing
          powerPreference: "high-performance", // Prefer GPU performance
          stencil: false, // Disable stencil buffer if not needed
          depth: true,
          logarithmicDepthBuffer: false, // Disable for better performance
        }}
        dpr={[1, 2]} // Limit device pixel ratio (1x to 2x max)
        performance={{ min: 0.5 }} // Allow frame drops below 50% performance
        style={{ background: "transparent", pointerEvents: "none" }}
      >
        {/* Simplified lighting for better performance */}
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 10, 5]} intensity={0.8} />
        <Suspense fallback={null}>
          <GLTFModel modelPath={modelPath} />
        </Suspense>
        {/* Use simpler environment or remove if not needed */}
        <Environment preset="night" />
      </Canvas>
    </div>
  );
}

// Spline embed component - uses Spline's embed URL
// Get your Spline scene URL from: https://spline.design
// After creating/publishing your scene, copy the embed URL
export function SplineModel({ sceneUrl }) {
  // Spline embed URL format: https://my.spline.design/[scene-name].scene
  // Or use their embed code format
  const splineUrl = sceneUrl || import.meta.env.VITE_SPLINE_SCENE_URL;

  if (
    !splineUrl ||
    splineUrl.includes("untitled") ||
    splineUrl.includes("example")
  ) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-black/20 rounded-lg border border-blue-500/20">
        <div className="text-center p-8">
          <div className="w-16 h-16 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400 mb-2 text-lg">3D Model Placeholder</p>
          <p className="text-xs text-gray-500 max-w-md">
            To add your Spline model:
            <br />
            1. Create a scene at{" "}
            <a
              href="https://spline.design"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-400 hover:underline"
            >
              spline.design
            </a>
            <br />
            2. Publish your scene and copy the embed URL
            <br />
            3. Add it to <code className="text-blue-300">.env</code> as{" "}
            <code className="text-blue-300">VITE_SPLINE_SCENE_URL</code>
            <br />
            Or pass it as <code className="text-blue-300">splineUrl</code> prop
          </p>
        </div>
      </div>
    );
  }

  // Spline embed iframe
  return (
    <iframe
      src={splineUrl}
      title="3D Model"
      className="w-full h-full border-0 rounded-lg"
      style={{ background: "transparent" }}
      allow="fullscreen"
      loading="lazy"
    />
  );
}

// Rive animation component
export function RiveAnimation({ src, artboard, stateMachines }) {
  const riveSrc = src || import.meta.env.VITE_RIVE_FILE_URL;

  if (!riveSrc || riveSrc.includes("example")) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-black/20 rounded-lg border border-blue-500/20">
        <div className="text-center p-8">
          <div className="w-16 h-16 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400 mb-2 text-lg">
            Rive Animation Placeholder
          </p>
          <p className="text-xs text-gray-500 max-w-md">
            To add your Rive animation:
            <br />
            1. Create an animation at{" "}
            <a
              href="https://rive.app"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-400 hover:underline"
            >
              rive.app
            </a>
            <br />
            2. Export your .riv file and host it
            <br />
            3. Add the URL to <code className="text-blue-300">
              .env
            </code> as <code className="text-blue-300">VITE_RIVE_FILE_URL</code>
            <br />
            Or pass it as <code className="text-blue-300">riveSrc</code> prop
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full">
      <Rive
        src={riveSrc}
        artboard={artboard}
        stateMachines={stateMachines}
        className="w-full h-full"
      />
    </div>
  );
}

// Main Hero3D component
// Usage:
// <Hero3D type="gltf" modelPath="/path/to/model.gltf" />
// <Hero3D type="spline" splineUrl="https://my.spline.design/your-scene.scene" />
// <Hero3D type="rive" riveSrc="https://your-cdn.com/animation.riv" />
export default function Hero3D({
  type = "gltf",
  modelPath = null,
  splineUrl = null,
  riveSrc = null,
  riveArtboard = null,
  riveStateMachines = null,
}) {
  if (type === "gltf") {
    return (
      <Suspense
        fallback={
          <div className="w-full h-full flex items-center justify-center">
            <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          </div>
        }
      >
        <GLTFViewer modelPath={modelPath} />
      </Suspense>
    );
  }

  if (type === "rive") {
    return (
      <Suspense
        fallback={
          <div className="w-full h-full flex items-center justify-center">
            <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          </div>
        }
      >
        <RiveAnimation
          src={riveSrc}
          artboard={riveArtboard}
          stateMachines={riveStateMachines}
        />
      </Suspense>
    );
  }

  // Default to Spline
  return (
    <Suspense
      fallback={
        <div className="w-full h-full flex items-center justify-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
        </div>
      }
    >
      <SplineModel sceneUrl={splineUrl} />
    </Suspense>
  );
}
