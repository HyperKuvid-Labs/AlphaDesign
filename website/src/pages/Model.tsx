import { Suspense, useRef, useState } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { OrbitControls, Environment, Text } from '@react-three/drei';
import { RotateCcw, ZoomIn, ZoomOut, Move3D } from 'lucide-react';
import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';

// Fallback component while loading
const LoadingComponent = () => (
  <mesh>
    <boxGeometry args={[2, 0.5, 1]} />
    <meshStandardMaterial color="#444444" wireframe />
  </mesh>
);

// STL Model component that loads the actual model file
const STLModelComponent = () => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  // Load the STL file
  const geometry = useLoader(STLLoader, '/model.stl');


  //Uncomment this for rotation
  // useFrame((state, delta) => {
  //   if (meshRef.current && !hovered) {
  //     meshRef.current.rotation.y += delta * 0.3;
  //   }
  // });

  // Center the geometry
  if (geometry) {
    geometry.computeBoundingBox();
    const center = new THREE.Vector3();
    geometry.boundingBox?.getCenter(center);
    geometry.translate(-center.x, -center.y, -center.z);
  }

  return (
    <group>
      <mesh
        ref={meshRef}
        geometry={geometry}
        position={[0, 0, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
        scale={hovered ? 0.021 : 0.02}
      >
        <meshStandardMaterial 
          color={hovered ? "#dc2626" : "#666666"} 
          metalness={0.7}
          roughness={0.3}
        />
      </mesh>
      
      {/* Model Label */}
      <Text
        position={[0, 3, 0]}
        fontSize={0.5}
        color="#ffffff"
        anchorX="center" 
        anchorY="middle"
      >
        ALPHA DESIGN F1 Model
      </Text>
    </group>
  );
};

const Model = () => {
  const [controlsRef, setControlsRef] = useState<any>(null);

  const resetCamera = () => {
    if (controlsRef) {
      controlsRef.reset();
    }
  };

  const zoomIn = () => {
    if (controlsRef) {
      controlsRef.dollyIn(1.2);
      controlsRef.update();
    }
  };

  const zoomOut = () => {
    if (controlsRef) {
      controlsRef.dollyOut(1.2);
      controlsRef.update();
    }
  };

  return (
    <div className="min-h-screen bg-background py-20">
      <div className="container mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-6xl font-rajdhani font-bold text-foreground mb-4">
            3D Model Viewer
          </h1>
          <p className="text-xl font-inter text-muted-foreground max-w-2xl mx-auto">
            Interactive 3D visualization of the Alpha-Design Formula 1 concept-(our most updateded one). 
            Drag to rotate, scroll to zoom, and explore every detail.
          </p>
        </div>

        {/* 3D Model Viewer */}
        <div className="max-w-6xl mx-auto">
          <div className="model-viewer h-[600px] md:h-[700px] relative">
            {/* Canvas */}
            <Canvas
              camera={{ position: [15, 8, 15], fov: 50 }}
              className="w-full h-full"
            >
              <Suspense fallback={<LoadingComponent />}>
                {/* Lighting */}
                <ambientLight intensity={0.4} />
                <directionalLight position={[10, 10, 5]} intensity={1} />
                <directionalLight position={[-10, -10, -5]} intensity={0.5} />
                <pointLight position={[0, 5, 0]} intensity={0.8} color="#dc2626" />
                
                {/* Environment */}
                <Environment preset="studio" />
                
                {/* 3D Model */}
                <STLModelComponent />
                
                {/* Controls */}
                <OrbitControls
                  ref={setControlsRef}
                  enablePan={true}
                  enableZoom={true}
                  enableRotate={true}
                  minDistance={3}
                  maxDistance={50}
                  maxPolarAngle={Math.PI / 2}
                />
              </Suspense>
            </Canvas>

            {/* Control Buttons */}
            <div className="absolute top-4 right-4 flex flex-col space-y-2">
              <button
                onClick={resetCamera}
                className="p-3 bg-card border border-border rounded-lg hover:bg-accent transition-racing"
                title="Reset View"
              >
                <RotateCcw className="w-5 h-5" />
              </button>
              <button
                onClick={zoomIn}
                className="p-3 bg-card border border-border rounded-lg hover:bg-accent transition-racing"
                title="Zoom In"
              >
                <ZoomIn className="w-5 h-5" />
              </button>
              <button
                onClick={zoomOut}
                className="p-3 bg-card border border-border rounded-lg hover:bg-accent transition-racing"
                title="Zoom Out"
              >
                <ZoomOut className="w-5 h-5" />
              </button>
            </div>

            {/* Instructions */}
            <div className="absolute bottom-4 left-4 bg-card/95 backdrop-blur-sm border border-border rounded-lg p-4 max-w-xs">
              <div className="flex items-center space-x-2 mb-2">
                <Move3D className="w-5 h-5 text-primary" />
                <span className="font-rajdhani font-semibold">Controls</span>
              </div>
              <div className="text-sm text-muted-foreground space-y-1">
                <div>• Drag to rotate</div>
                <div>• Scroll to zoom</div>
                <div>• Right-click to pan</div>
              </div>
            </div>
          </div>

          {/* Model Information */}
          <div className="mt-12 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="racing-card p-6 text-center">
              <h3 className="text-xl font-rajdhani font-semibold mb-3 text-primary">
                Main Specifications
              </h3>
              <div className="space-y-2 text-muted-foreground">
                <div>Total span : 1600</div>
                <div>Tip chord : 250</div>
                <div>Flap count : 3</div>
                <div>Pylon count : 2</div>
              </div>
            </div>

            <div className="racing-card p-6 text-center">
              <h3 className="text-xl font-rajdhani font-semibold mb-3 text-secondary">
                Endplate specifications
              </h3>
              <div className="space-y-2 text-muted-foreground">
                <div>Endplate height : 280</div>
                <div>Endplate max_width : 120</div>
                <div>Endplate thickness_base : 10</div>
                <div>Endplate forward_lean : 6</div>
              </div>
            </div>

            <div className="racing-card p-6 text-center md:col-span-2 lg:col-span-1">
              <h3 className="text-xl font-rajdhani font-semibold mb-3 text-metallic">
                Target and materials
              </h3>
              <div className="space-y-2 text-muted-foreground">
                <div>Material : Standard Carbon Fiber</div>
                <div>Target Downforce : 4000</div>
                <div>Target Drag : 40</div>
                <div>Efficiency Factor : 1.0</div>
              </div>
            </div>
          </div>

          {/* Note about STL */}
          {/* <div className="mt-8 text-center">
            <p className="text-sm text-muted-foreground font-inter">
              * This is a conceptual 3D model. The actual STL file will be loaded from /public/model.stl when available.
            </p>
          </div> */}
        </div>
      </div>
    </div>
  );
};

export default Model;