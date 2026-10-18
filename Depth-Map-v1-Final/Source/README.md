# Depth Map v1 - Handover Documentation

## 🚀 Project Overview
Depth Map v1 is a high-performance, native macOS application designed for Al-powered depth estimation. Built specifically for Apple Silicon, it leverages the **Depth Anything V2** model to generate high-fidelity depth maps from 2D images with millisecond latency.

### Key Features
- **AI Depth Inference**: Real-time depth map generation using CoreML.
- **Parallax Magnifier**: A dynamic inspection tool that zooms and simulates 3D depth parallax as you move your cursor.
- **Premium UI System**: Custom-built `ModernSlider` and `ModernToggle` components for a sleek, unified aesthetic.
- **Modular Post-Processing**: Independent filters for Blur, Sharpness, Brightness, and Contrast, each color-coded with specific icons.
- **Dual Language Support**: Full Turkish and English localization (TR/EN).
- **Pro Export**: Direct export to high-quality image formats with post-processing applied.

---

## 🛠 Technical Architecture
- **Language**: Swift 6.0
- **UI Framework**: SwiftUI (Native Metal-backed)
- **AI Engine**: CoreML (Optimized for Neural Engine)
- **Model**: Depth Anything V2 (Small-F16 version for speed/accuracy balance)
- **Minimum OS**: macOS 14.1 (Sonoma) or newer

---

## 📦 File Structure
- `main.swift`: The entire application logic, UI components, and localization database.
- `DepthAnythingV2SmallF16.mlmodelc/`: The compiled CoreML model folder.
- `MyIcon.icns`: The high-resolution application icon.
- `package_dmg.sh`: Automated bash script for building the App bundle and DMG installer.

---

## 🔨 Developer Build Guide
To recompile the application or package a new DMG, use the following commands from the terminal:

### 1. Simple Compilation (Fast Test)
```bash
swiftc -sdk $(xcrun --show-sdk-path --sdk macosx) -target arm64-apple-macos14.0 -parse-as-library -o "Depth Map v1.app/Contents/MacOS/Depth Map v1" main.swift
```

### 2. Full Packaging (DMG Creation)
```bash
chmod +x package_dmg.sh
./package_dmg.sh
```

---

## 🛤 Future Roadmap (Suggested Improvements)
If the project is advanced further, consider these features:

1. **Video Depth Processing**: Implement batch processing or real-time video depth estimation using `AVFoundation`.
2. **3D Point Cloud Export**: Export depth data as `.obj` or `.stl` files for 3D printing and CAD software.
3. **Model Selection**: Allow users to toggle between "Small", "Base", and "Large" models depending on their hardware (M1 vs M3 Max).
4. **iCloud Integration**: Sync processed depth maps across Apple devices.
5. **ARKit Preview**: Enable a way to view the 3D depth results in augmented reality via USDZ files.

---

## ⚖️ License
This project is licensed under the MIT License. AI Model weights are subject to the original Depth Anything V2 licensing terms.

**Developed with ❤️ for Depth Map v1.**
