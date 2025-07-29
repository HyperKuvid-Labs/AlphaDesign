# Complete F1 Front Wing Technical Specification
## Ultra-Detailed STL Modeling Guide (2022-2025 Regulations)

### Document Version: 2.1
### Target: Script-based STL generation for accurate F1 front wing modeling
### References: 2025 F1 season, Drive to Survive S6, F1 movie (2024), technical imagery

---

## 1. MAIN WING STRUCTURE (Primary Element)

### 1.1 Main Plane Geometry
- **Total Span (width):** 1,800mm ±5mm (FIA maximum width)
- **Root Chord (centerline):** 305mm ±10mm
- **Tip Chord (at endplate):** 280mm ±8mm
- **Chord Taper Ratio:** 0.918 (linear transition)
- **Sweep Angle:** 4.5° ±1° (measured at quarter-chord line)
- **Dihedral Angle:** 3.2° ±0.8° (tips higher than center)
- **Twist Distribution:** -2° to +1° (nose-up at tips)

### 1.2 Airfoil Profile Details
- **Base Profile:** Modified inverted NACA 64A010
- **Maximum Thickness:** 18.5mm at 35% chord from leading edge
- **Camber:** 10.8% at 42% chord position
- **Leading Edge Radius:** 3.2mm (constant span-wise)
- **Trailing Edge Thickness:** 2.1mm (STL minimum: 2.5mm)
- **Upper Surface Curvature:** Concave with R=850mm primary radius
- **Lower Surface Curvature:** Convex with R=1200mm primary radius

### 1.3 Structural Features
- **Mounting Boss Locations:** ±180mm from centerline
- **Boss Diameter:** 45mm with 8mm center hole
- **Nose Integration Fillet:** 32mm radius, blended over 120mm length
- **Reinforcement Ribs:** Internal, 4mm thick, every 150mm span-wise
- **Drain Holes:** 3mm diameter, lowest point of each rib bay
- **Wall Thickness:** 
  - Leading edge: 8mm
  - Main body: 5mm minimum
  - Trailing edge: 3mm (STL: 3.5mm minimum)

---

## 2. FLAP SYSTEM (Multi-Element Configuration)

### 2.1 Flap Count and Arrangement
- **Total Flaps:** 4 (standard 2022-2025 configuration)
- **Numbering:** F1 (lowest/forward) to F4 (highest/rearward)
- **Span Variation:** Stepped reduction outboard for Y250 compliance

### 2.2 Individual Flap Specifications

#### Flap 1 (Primary/Lowest)
- **Span:** 1,750mm (full width minus endplate thickness)
- **Root Chord:** 245mm
- **Tip Chord:** 220mm
- **Profile:** Custom high-camber (14.2% at 38% chord)
- **Slot Gap from Main:** 16mm ±1mm
- **Vertical Offset:** 28mm above main plane trailing edge
- **Horizontal Offset:** 35mm aft of main plane trailing edge
- **Mounting Points:** 6 per side, integrated tabs

#### Flap 2 (Secondary)
- **Span:** 1,680mm
- **Root Chord:** 195mm
- **Tip Chord:** 175mm
- **Profile:** Medium camber (11.8% at 40% chord)
- **Slot Gap from F1:** 14mm ±0.8mm
- **Vertical Offset:** 52mm above main plane
- **Horizontal Offset:** 68mm aft of main plane
- **End Treatment:** Rounded tips with 8mm radius

#### Flap 3 (Tertiary)
- **Span:** 1,580mm
- **Root Chord:** 165mm
- **Tip Chord:** 145mm
- **Profile:** Lower camber (9.2% at 42% chord)
- **Slot Gap from F2:** 12mm ±0.6mm
- **Vertical Offset:** 78mm above main plane
- **Horizontal Offset:** 95mm aft of main plane
- **Adjustment Range:** ±15mm (if adjustable variant)

#### Flap 4 (Top/Gurney)
- **Span:** 1,450mm
- **Root Chord:** 125mm
- **Tip Chord:** 110mm
- **Profile:** Low camber (6.8% at 45% chord)
- **Slot Gap from F3:** 10mm ±0.5mm
- **Vertical Offset:** 105mm above main plane
- **Horizontal Offset:** 118mm aft of main plane
- **Gurney Height:** 4-8mm (team-specific)

### 2.3 Flap Connection System
- **Tab Style:** T-slot with 0.2mm clearance
- **Material Bridges:** 2mm thick, every 80mm
- **Fastener Holes:** M4 thread (4.2mm STL hole)
- **Alignment Pins:** 2mm diameter, 8mm depth
- **Assembly Sequence:** Main → F1 → F2 → F3 → F4

---

## 3. ENDPLATE SYSTEM (Aerodynamic Walls)

### 3.1 Overall Endplate Geometry
- **Height:** 325mm (from main plane lower surface)
- **Maximum Width:** 135mm (at main plane junction)
- **Minimum Width:** 45mm (at top edge)
- **Thickness Profile:** 
  - Base: 12mm
  - Mid-height: 8mm
  - Top edge: 5mm
  - Minimum STL: 3mm anywhere

### 3.2 Complex 3D Curvature
- **Primary Curve:** S-shape in side view
  - Forward lean at top: 8° from vertical
  - Rearward sweep at bottom: 12° from vertical
- **Outboard Wrap:** 22° inward turn (toward wheel well)
- **Top Edge Sculpting:** Wavy profile, amplitude ±8mm
- **Transition Radii:**
  - To main plane: 25mm
  - To ground plane: 18mm
  - Internal corners: 6mm minimum

### 3.3 Footplate and Lower Features
- **Footplate Extension:** 85mm forward of main plane leading edge
- **Footplate Height:** 35mm above reference plane
- **Arch Radius:** 145mm (creates tunnel under endplate)
- **Footplate Thickness:** 6mm (constant)
- **Drainage Scallops:** 3 per side, 12mm radius

### 3.4 Strakes and Auxiliary Elements
- **Primary Strakes:** 2 per endplate
  - Height: 55mm and 42mm
  - Thickness: 3mm
  - Location: Bottom third of endplate
  - Angle: 15° upward from horizontal
- **Vortex Generators:** 4-6 small bumps
  - Height: 2-4mm
  - Diameter: 8-15mm
  - Location: Upper rear quadrant
- **Slots/Cutouts:** Team-specific
  - Width: 5-25mm
  - Purpose: Pressure relief and flow management

### 3.5 Mounting and Integration
- **Main Plane Attachment:** 
  - 8 bolt holes: M5 (5.2mm STL)
  - Spacing: 35mm centers
  - Countersink: 2mm depth, 10mm diameter
- **Flap Attachment Points:**
  - F1: 4 tabs per endplate
  - F2-F4: 2 tabs each
  - Tab thickness: 4mm
  - Tab width: 18mm

---

## 4. Y250 VORTEX REGION (Central Section)

### 4.1 Regulatory Geometry
- **Width Definition:** 250mm each side of centerline (500mm total)
- **Height Restriction:** Must be lower than outboard sections
- **Step Height:** 15-25mm below outboard flap level
- **Transition Length:** 100mm blend zone at 250mm stations

### 4.2 Specific Features
- **Central Slot:** 35mm wide, extends through all elements
- **Neutral Section:** Reduced camber airfoils in central 200mm
- **Vortex Initiation:** Sharp edge at 250mm boundary
- **Flow Conditioning:** Small strakes (2-3mm high) on upper surface
- **Mounting Clearance:** Extra space for nose attachment hardware

---

## 5. MOUNTING PYLON SYSTEM (Nose Integration)

### 5.1 Pylon Configuration
- **Count:** 2 primary pylons (current regulations)
- **Spacing:** 360mm center-to-center
- **Section Shape:** Elliptical, streamlined
- **Major Axis:** 42mm (horizontal)
- **Minor Axis:** 28mm (vertical)
- **Length:** 140mm (nose to wing)

### 5.2 Aerodynamic Integration
- **Nose Fillet Radius:** 35mm all around
- **Wing Junction Radius:** 28mm
- **Flow Transition:** NACA duct-style blending
- **Anti-Vibration:** Internal stiffening ribs
- **Access Holes:** For internal wiring/sensors (if applicable)

### 5.3 Structural Details
- **Material Thickness:** 6mm minimum
- **Internal Reinforcement:** Cross-bracing every 40mm
- **Bolt Attachments:**
  - Nose end: 4x M6 bolts
  - Wing end: 6x M5 bolts
- **Alignment Features:** Locating pins and reference surfaces

---

## 6. SLOT GAP MANAGEMENT (Aerodynamic Channels)

### 6.1 Gap Specifications
- **Gap 1 (Main to F1):** 16mm ±0.5mm
- **Gap 2 (F1 to F2):** 14mm ±0.5mm  
- **Gap 3 (F2 to F3):** 12mm ±0.5mm
- **Gap 4 (F3 to F4):** 10mm ±0.5mm

### 6.2 Gap Profile Management
- **Entry Angle:** 5-8° divergent
- **Exit Angle:** 2-4° convergent
- **Throat Width:** Minimum gap dimension
- **Wall Smoothness:** Ra 0.4μm equivalent (STL: smooth curves)
- **Edge Radii:** 1.5mm minimum on all gap edges

### 6.3 Flow Control Features
- **Slot Lips:** 1-2mm thickness extensions
- **Pressure Equalization:** Small holes (1mm) every 100mm
- **Turbulence Management:** Micro-vortex generators (0.5mm high)

---

## 7. CASCADE ELEMENTS AND AUXILIARY DEVICES

### 7.1 Cascade Winglets
- **Primary Cascade:**
  - Span: 280mm
  - Chord: 65mm
  - Thickness: 8mm
  - Location: Forward of endplate, above main plane
  - Angle: 35° to main plane
  - Profile: Symmetrical NACA 0008

- **Secondary Cascade:**
  - Span: 180mm
  - Chord: 45mm
  - Thickness: 6mm
  - Location: Mid-height on endplate
  - Angle: 25° to vertical
  - Profile: Cambered 4%

### 7.2 Dive Planes and Strakes
- **Dive Plane Count:** 1-2 per endplate (team variant)
- **Dimensions:** 85mm × 35mm × 4mm
- **Location:** Lower forward edge of endplate
- **Angle:** 12° downward from horizontal
- **Mounting:** Integral with endplate structure

### 7.3 Vortex Management Devices
- **Micro-Strakes:** 15-25 per wing
  - Height: 1-3mm
  - Length: 8-20mm
  - Thickness: 0.8-1.5mm
  - Location: Element intersections and high-gradient areas

---

## 8. MATERIALS AND MANUFACTURING (STL Optimization)

### 8.1 Wall Thickness Standards
- **Structural Areas:** 5-12mm
- **Aerodynamic Surfaces:** 3-8mm
- **Detail Features:** 2-4mm
- **Absolute Minimum (STL):** 2.5mm anywhere
- **Connection Points:** 8-15mm for strength

### 8.2 Fillet and Radius Requirements
- **External Fillets:** 8-25mm (aerodynamic)
- **Internal Fillets:** 3-8mm (stress relief)
- **Detail Radii:** 1-3mm (printability)
- **Sharp Edges:** Avoided (minimum 0.5mm radius)

### 8.3 Assembly Features
- **Pin Holes:** 
  - Diameter: 3mm, 4mm, 5mm (standard sizes)
  - Tolerance: +0.2mm for clearance fit
  - Depth: 1.5× diameter minimum
  - Chamfer: 0.5mm × 45°

- **Bolt Holes:**
  - M3: 3.2mm diameter
  - M4: 4.2mm diameter  
  - M5: 5.2mm diameter
  - M6: 6.2mm diameter
  - Countersink: 2mm depth standard

### 8.4 Mesh Resolution Guidelines
- **Aerodynamic Surfaces:** 0.3mm triangle edge length
- **Structural Areas:** 0.5mm triangle edge length
- **Detail Features:** 0.2mm triangle edge length
- **Hidden Surfaces:** 0.8mm triangle edge length

---

## 9. REGULATORY COMPLIANCE (FIA 2022-2025)

### 9.1 Dimensional Limits
- **Maximum Width:** 1,800mm
- **Maximum Height:** 330mm above reference plane
- **Y250 Restriction:** Central 500mm must be stepped down
- **Legality Box:** All geometry within defined 3D envelope
- **Minimum Radii:** 5mm on all external edges

### 9.2 Load Test Considerations
- **Deflection Limits:** <5mm under standard loads
- **Mounting Points:** Must withstand 2,000N vertical loads
- **Safety Factors:** 2.5× working loads minimum
- **Failure Modes:** Controlled breakaway designed in

---

## 10. TEAM-SPECIFIC VARIATIONS (2025 Season)

### 10.1 Mercedes W16 Style
- **Endplate Shape:** More curved, tighter wrap
- **Flap Count:** 4 with thin profiles
- **Cascade:** Prominent double-element
- **Color Breaks:** Silver/black/teal scheme reference

### 10.2 Red Bull RB21 Style  
- **Endplate Shape:** Angular, aggressive footplate
- **Flap Count:** 4 with high camber
- **Y250 Treatment:** Sharp vortex initiation
- **Color Breaks:** Navy/red/yellow scheme reference

### 10.3 Ferrari SF-25 Style
- **Endplate Shape:** Smooth curves, elegant transitions
- **Flap Count:** 4 with medium camber
- **Strake Details:** Subtle, integrated design
- **Color Breaks:** Red/white/yellow scheme reference

---

## 11. SCRIPT IMPLEMENTATION EXAMPLE

