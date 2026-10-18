import SwiftUI
import CoreML
import Vision
import UniformTypeIdentifiers
import CoreImage.CIFilterBuiltins

enum AppLanguage: String, CaseIterable, Identifiable {
    case tr = "Türkçe"
    case en = "English"
    var id: Self { self }
}

func loc(_ key: String, _ lang: AppLanguage) -> String {
    let strings: [String: [AppLanguage: String]] = [
        "app_title": [.tr: "Depth Map v1", .en: "Depth Map v1"],
        "fine_tuning": [.tr: "İnce Ayarlar", .en: "Fine Tuning"],
        "language": [.tr: "Dil / Language", .en: "Dil / Language"],
        "model_hw": [.tr: "Model & Donanım", .en: "Model & Hardware"],
        "compute_unit": [.tr: "İşlem Birimi", .en: "Compute Unit"],
        "image_proc": [.tr: "Görüntü İşleme", .en: "Image Processing"],
        "color_palette": [.tr: "Renk Paleti", .en: "Color Palette"],
        "scale_mode": [.tr: "Sığdırma Modu", .en: "Scale Mode"],
        "orig_image": [.tr: "Orijinal Resim", .en: "Original Image"],
        "click_to_select": [.tr: "Seçmek veya Sürüklemek İçin", .en: "Click or Drop to Select"],
        "depth_map": [.tr: "Derinlik Haritası", .en: "Depth Map"],
        "processing": [.tr: "İşleniyor...", .en: "Processing..."],
        "waiting": [.tr: "Bekleniyor...", .en: "Waiting..."],
        "neural_proc": [.tr: "Nöral Motor'da İşleniyor...", .en: "Processing in Neural Engine..."],
        "save_result": [.tr: "Sonucu Kaydet", .en: "Save Result"],
        "drop_zone": [.tr: "Buraya Bırakın", .en: "Drop Here"],
        "post_processing": [.tr: "Derinlik Filtreleri", .en: "Post-Processing"],
        "blur": [.tr: "Yumuşatma", .en: "Blur"],
        "sharpness": [.tr: "Keskinleştirme", .en: "Sharpness"],
        "brightness": [.tr: "Parlaklık", .en: "Brightness"],
        "contrast": [.tr: "Kontrast", .en: "Contrast"],
        "magnifier": [.tr: "Büyüteç Aracı", .en: "Magnifier Tool"],
        "about_app": [.tr: "Hakkında Depth Map", .en: "About Depth Map"],
        "file_open": [.tr: "Görsel Yükle", .en: "Open Image"],
        "file_save": [.tr: "Sonucu Kaydet", .en: "Save Result"],
        "copyright": [.tr: "© 2026 gurkanerol Tüm Hakları Saklıdır.", .en: "© 2026 gurkanerol All Rights Reserved."],
        "created_by": [.tr: "Created by gurkanerol", .en: "Created by gurkanerol"],
        "help_menu": [.tr: "Kullanım Kılavuzu", .en: "User Guide"]
    ]
    return strings[key]?[lang] ?? key
}

enum ColorMode: String, CaseIterable, Identifiable {
    case grayscale
    case thermal
    var id: Self { self }
    
    func localized(lang: AppLanguage) -> String {
        switch self {
        case .grayscale: return lang == .tr ? "Siyah & Beyaz" : "Grayscale"
        case .thermal: return lang == .tr ? "Termal" : "Thermal"
        }
    }
}

enum ComputeUnitOption: String, CaseIterable, Identifiable {
    case all
    case cpuAndGPU
    case cpuOnly
    var id: Self { self }
    
    func localized(lang: AppLanguage) -> String {
        switch self {
        case .all: return lang == .tr ? "Tümü (GPU+NPU+CPU)" : "All (GPU+NPU+CPU)"
        case .cpuAndGPU: return lang == .tr ? "Sadece CPU & GPU" : "CPU & GPU Only"
        case .cpuOnly: return lang == .tr ? "Sadece CPU" : "CPU Only"
        }
    }
    
    var mlComputeUnits: MLComputeUnits {
        switch self {
        case .all: return .all
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuOnly: return .cpuOnly
        }
    }
}

enum CropOption: String, CaseIterable, Identifiable {
    case scaleFill
    case scaleFit
    case centerCrop
    var id: Self { self }
    
    func localized(lang: AppLanguage) -> String {
        switch self {
        case .scaleFill: return lang == .tr ? "Esneterek Sığdır" : "Scale Fill"
        case .scaleFit: return lang == .tr ? "Oranlı Sığdır" : "Scale Fit"
        case .centerCrop: return lang == .tr ? "Ortadan Kes" : "Center Crop"
        }
    }
    
    var vnOption: VNImageCropAndScaleOption {
        switch self {
        case .scaleFill: return .scaleFill
        case .scaleFit: return .scaleFit
        case .centerCrop: return .centerCrop
        }
    }
}

class DepthModelManager: ObservableObject {
    @Published var outputImage: NSImage?
    @Published var isProcessing: Bool = false
    @Published var isFiltering: Bool = false
    
    var vnModel: VNCoreMLModel?
    
    private var rawModelResultCIImage: CIImage?
    private var targetResolution: CGSize = .zero
    
    init() {
        loadModel(computeUnits: .all)
    }
    
    func loadModel(computeUnits: MLComputeUnits) {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        
        var url = Bundle.main.url(forResource: "DepthAnythingV2SmallF16", withExtension: "mlmodelc")
        if url == nil {
            url = URL(fileURLWithPath: "DepthAnythingV2SmallF16.mlmodelc")
        }
        
        guard let validUrl = url else {
            print("Model bulunamadı!")
            return
        }
        
        do {
            let mlModel = try MLModel(contentsOf: validUrl, configuration: config)
            self.vnModel = try VNCoreMLModel(for: mlModel)
            print("Model başarıyla yüklendi. Compute Units: \(computeUnits.rawValue)")
        } catch {
            print("Model yükleme hatası: \(error)")
        }
    }
    
    func predict(image: NSImage, scaleOption: VNImageCropAndScaleOption,
                 colorMode: ColorMode, blur: Double, sharpness: Double, brightness: Double, contrast: Double) {
        guard let vnModel = vnModel else { return }
        
        DispatchQueue.main.async {
            self.outputImage = nil
            self.isProcessing = true
        }
        
        DispatchQueue.global(qos: .userInitiated).async {
            guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }
            
            let request = VNCoreMLRequest(model: vnModel) { [weak self] request, error in
                guard let self = self else { return }
                
                self.targetResolution = CGSize(width: cgImage.width, height: cgImage.height)
                
                func handleBuffer(_ buffer: CVPixelBuffer) {
                    self.rawModelResultCIImage = CIImage(cvPixelBuffer: buffer)
                    self.applyFilters(colorMode: colorMode, blur: blur, sharpness: sharpness, brightness: brightness, contrast: contrast, completion: {
                        DispatchQueue.main.async { self.isProcessing = false }
                    })
                }
                
                if let results = request.results as? [VNCoreMLFeatureValueObservation],
                   let first = results.first,
                   let pixelBuffer = first.featureValue.imageBufferValue {
                    handleBuffer(pixelBuffer)
                } else if let results = request.results as? [VNPixelBufferObservation],
                          let first = results.first {
                     handleBuffer(first.pixelBuffer)
                } else {
                     DispatchQueue.main.async { self.isProcessing = false }
                }
            }
            
            request.imageCropAndScaleOption = scaleOption
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            try? handler.perform([request])
        }
    }
    
    func applyFilters(colorMode: ColorMode, blur: Double, sharpness: Double, brightness: Double, contrast: Double, completion: (() -> Void)? = nil) {
        guard var ciImage = rawModelResultCIImage, targetResolution != .zero else {
            completion?()
            return
        }
        
        DispatchQueue.main.async {
            self.isFiltering = true
        }
        
        DispatchQueue.global(qos: .userInitiated).async {
            let scaleX = self.targetResolution.width / ciImage.extent.width
            let scaleY = self.targetResolution.height / ciImage.extent.height
            ciImage = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            let targetRect = CGRect(x: 0, y: 0, width: self.targetResolution.width, height: self.targetResolution.height)
            ciImage = ciImage.cropped(to: targetRect)
            
            let colorFilter = CIFilter.colorControls()
            colorFilter.inputImage = ciImage
            colorFilter.brightness = Float(brightness)
            colorFilter.contrast = Float(contrast)
            if let out = colorFilter.outputImage { ciImage = out }
            
            if blur > 0.0 {
                let blurFilter = CIFilter.gaussianBlur()
                blurFilter.inputImage = ciImage
                blurFilter.radius = Float(blur)
                if let out = blurFilter.outputImage { ciImage = out }
            }
            
            if sharpness > 0.0 {
                let sharpFilter = CIFilter.unsharpMask()
                sharpFilter.inputImage = ciImage
                sharpFilter.intensity = Float(sharpness)
                sharpFilter.radius = 2.5
                if let out = sharpFilter.outputImage { ciImage = out }
            }
            
            ciImage = ciImage.cropped(to: targetRect)
            
            if colorMode == .thermal {
                let thermalFilter = CIFilter.falseColor()
                thermalFilter.inputImage = ciImage
                thermalFilter.color0 = CIColor(red: 0.0, green: 0.0, blue: 0.6)
                thermalFilter.color1 = CIColor(red: 1.0, green: 0.2, blue: 0.0)
                if let out = thermalFilter.outputImage { ciImage = out }
            }
            
            let context = CIContext(options: [.highQualityDownsample: true])
            if let cgOutput = context.createCGImage(ciImage, from: targetRect) {
                let outImage = NSImage(cgImage: cgOutput, size: self.targetResolution)
                DispatchQueue.main.async {
                    self.outputImage = outImage
                    self.isFiltering = false
                    completion?()
                }
            } else {
                DispatchQueue.main.async {
                    self.isFiltering = false
                    completion?()
                }
            }
        }
    }
}

// YENİ EKLENTİ: Fareyi takip eden interaktif sanal büyüteç katmanı
struct MagnifiableImageView: View {
    var image: NSImage
    var isEnabled: Bool
    @Binding var zoom: Double
    
    @State private var hoverLocation: CGPoint? = nil
    
    // Büyüteç ayarları
    let lensSize: CGFloat = 180
    
    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .topLeading) {
                
                // Arkaplandaki normal resim
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
                    .cornerRadius(8)
                    .frame(width: proxy.size.width, height: proxy.size.height)
                
                // Üst katman: Büyüteç Lensi
                if isEnabled, let location = hoverLocation {
                    let clampedX = min(max(location.x, lensSize / 2), proxy.size.width - lensSize / 2)
                    let clampedY = min(max(location.y, lensSize / 2), proxy.size.height - lensSize / 2)
                    
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFit()
                        .cornerRadius(8) // Sadece resim sınırlarını kıvırır (Bütünlük kuralı)
                        // Öncelikle alt resimle birebir aynı ebatta çiziyoruz.
                        .frame(width: proxy.size.width, height: proxy.size.height)
                        // 1. Doğrudan farenin işaret ettiği "o piksel" etrafında resmi büyütüyoruz (Optik Merkez).
                        .scaleEffect(zoom, anchor: UnitPoint(x: location.x / proxy.size.width, y: location.y / proxy.size.height))
                        // 2. Optik noktayı, merceğin bulunduğu koordinata yerleştiriyoruz (Ekran kenarlarında parallax sağlar).
                        .offset(x: clampedX - location.x, y: clampedY - location.y)
                        // 3. Mercek boyutunda yuvarlak bir delik açıp sadece dışarıyı oradan izliyoruz.
                        .mask(
                            Circle()
                                .frame(width: lensSize, height: lensSize)
                                .position(x: clampedX, y: clampedY)
                        )
                        // 4. Dekoratif çerçeveyi çiziyoruz.
                        .overlay(
                            Circle()
                                .stroke(Color.white.opacity(0.8), lineWidth: 3)
                                .frame(width: lensSize, height: lensSize)
                                .position(x: clampedX, y: clampedY)
                                .shadow(color: .black.opacity(0.5), radius: 6, x: 0, y: 3)
                        )
                        .allowsHitTesting(false)
                }
            }
            .contentShape(Rectangle()) // Boşluklarda da farenin takip edilebilmesi için
            .onContinuousHover { phase in
                if isEnabled {
                    switch phase {
                    case .active(let location):
                        hoverLocation = location
                    case .ended:
                        hoverLocation = nil
                    }
                }
            }
        }
    }
}

// MARK: - Modern Slider Component
struct ModernSlider: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    var step: Double? = nil
    var format: String = "%.1f"
    var accentColor: Color = .blue
    var icon: String? = nil
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                if let icon = icon {
                    Image(systemName: icon)
                        .font(.system(size: 10))
                        .foregroundColor(accentColor)
                }
                Text(label)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.secondary)
                Spacer()
                Text("\(value, specifier: format)")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(.secondary)
            }
            
            GeometryReader { geometry in
                let width = geometry.size.width
                let height = geometry.size.height
                let thumbWidth: CGFloat = 28
                let percentage = CGFloat((value - range.lowerBound) / (range.upperBound - range.lowerBound))
                let clampedPercentage = max(0, min(1, percentage))
                
                ZStack(alignment: .leading) {
                    // Track Background
                    Capsule()
                        .fill(Color.black.opacity(0.3))
                        .overlay(
                            Capsule()
                                .stroke(Color.white.opacity(0.1), lineWidth: 0.5)
                        )
                    
                    // Fill (Custom Color)
                    Capsule()
                        .fill(accentColor)
                        .frame(width: clampedPercentage * width)
                    
                    // Symbols
                    HStack {
                        Image(systemName: "minus")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.white.opacity(0.5))
                            .padding(.leading, 8)
                        
                        Spacer()
                        
                        Image(systemName: "plus")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.white.opacity(0.5))
                            .padding(.trailing, 8)
                    }
                    
                    // Thumb
                    Capsule()
                        .fill(Color.white)
                        .frame(width: thumbWidth, height: height - 4)
                        .shadow(color: .black.opacity(0.3), radius: 2, x: 0, y: 1)
                        .offset(x: clampedPercentage * (width - thumbWidth))
                        .padding(.horizontal, 2)
                }
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { gesture in
                            let newValue = Double(gesture.location.x / width) * (range.upperBound - range.lowerBound) + range.lowerBound
                            var finalValue = max(range.lowerBound, min(range.upperBound, newValue))
                            if let step = step {
                                finalValue = (finalValue / step).rounded() * step
                            }
                            value = finalValue
                        }
                )
            }
            .frame(height: 22)
        }
    }
}

// MARK: - About View
struct AboutView: View {
    var language: AppLanguage
    @Binding var isPresented: Bool
    
    var body: some View {
        VStack(spacing: 20) {
            if let logoPath = Bundle.main.path(forResource: "logo", ofType: "png"),
               let logoImage = NSImage(contentsOfFile: logoPath) {
                Image(nsImage: logoImage)
                    .resizable()
                    .scaledToFit()
                    .frame(height: 60)
            } else {
                Text(loc("app_title", language))
                    .font(.largeTitle)
                    .bold()
            }
            
            Text("Version 1.0")
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            VStack(spacing: 5) {
                Text(loc("created_by", language))
                    .font(.headline)
                Text(loc("copyright", language))
                    .font(.caption)
                    .foregroundColor(.gray)
            }
            .padding(.vertical, 10)
            
            Button("OK") {
                isPresented = false
            }
            .buttonStyle(.borderedProminent)
            .keyboardShortcut(.defaultAction)
            .frame(width: 100)
        }
        .padding(40)
        .frame(width: 400)
    }
}

// MARK: - App Manual / Help View
struct HelpView: View {
    var language: AppLanguage
    @Binding var isPresented: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Text(language == .tr ? "Kullanım Kılavuzu" : "User Guide")
                    .font(.largeTitle)
                    .bold()
                Spacer()
                Button("X") { isPresented = false }
                    .buttonStyle(.borderless)
                    .font(.title2)
            }
            .padding(.bottom, 10)
            
            ScrollView {
                VStack(alignment: .leading, spacing: 25) {
                    helpSection(
                        icon: "photo.on.rectangle",
                        title: language == .tr ? "1. Görsel Yükleme" : "1. Loading Image",
                        desc: language == .tr ? "Görsellerinizi sol alana sürükleyip bırakabilir veya üst menüden (File > Open Image / Cmd+O) yükleyebilirsiniz." : "Drag and drop your images into the left area or use the top menu (File > Open Image / Cmd+O)."
                    )
                    
                    helpSection(
                        icon: "cpu",
                        title: language == .tr ? "2. Derinlik Analizi" : "2. Depth Analysis",
                        desc: language == .tr ? "Apple Neural Engine (NPU) kullanılarak fotoğrafınızın 3 boyutlu derinlik haritası saniyeler içinde çıkartılır." : "Your photo's 3D depth map is extracted directly on-device using the Apple Neural Engine (NPU)."
                    )
                    
                    helpSection(
                        icon: "slider.horizontal.3",
                        title: language == .tr ? "3. İnce Ayarlar" : "3. Fine-Tuning",
                        desc: language == .tr ? "Filtre menüsünden (Alt Kısım) haritanın bulanıklığını, keskinliğini veya kontrastını anlık olarak değiştirerek çıktıyı ihtiyaçlarınıza göre şekillendirebilirsiniz." : "Fine-tune the output by instantly changing the blur, sharpness, or contrast of the map from the Filter menu."
                    )

                    helpSection(
                        icon: "magnifyingglass",
                        title: language == .tr ? "4. Büyüteç Aracı" : "4. Magnifier Tool",
                        desc: language == .tr ? "Sağ üst köşedeki menüden (≡) Büyüteci aktif edebilirsiniz. Farenizi resmin üzerinde gezdirerek köşe detaylarını (paralaks mercekle) inceleyebilir, yakınlaştırma düzeyini ayarlayabilirsiniz." : "Activate the Magnifier from the top right menu (≡). Move your mouse over the result to inspect parallax details and adjust zoom level."
                    )
                    
                    helpSection(
                        icon: "square.and.arrow.down",
                        title: language == .tr ? "5. Dışa Aktarma" : "5. Exporting",
                        desc: language == .tr ? "Analiz bittiğinde 'Sonucu Kaydet' (Cmd+S) butonuna tıklayarak yüksek çözünürlüklü derinlik haritasını PNG olarak cihazınıza kaydedin." : "When the analysis is complete, click 'Save Result' (Cmd+S) to export the depth map to your device as a PNG."
                    )
                }
                .padding(.trailing, 10)
            }
            
            HStack {
                Spacer()
                Button(language == .tr ? "Anladım" : "Got it") {
                    isPresented = false
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
            .padding(.top, 10)
        }
        .padding(30)
        .frame(width: 550, height: 600)
    }
    
    func helpSection(icon: String, title: String, desc: String) -> some View {
        HStack(alignment: .top, spacing: 15) {
            Image(systemName: icon)
                .font(.system(size: 28))
                .foregroundColor(.accentColor)
                .frame(width: 45)
            VStack(alignment: .leading, spacing: 5) {
                Text(title).font(.title3).bold()
                Text(desc).font(.body).foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

struct ContentView: View {
    @StateObject private var depthManager = DepthModelManager()
    @State private var inputImage: NSImage?
    
    @AppStorage("appLang") private var appLangStr: String = AppLanguage.tr.rawValue
    private var language: AppLanguage { AppLanguage(rawValue: appLangStr) ?? .tr }
    
    @State private var colorMode: ColorMode = .grayscale
    @State private var computeUnit: ComputeUnitOption = .all
    @State private var scaleOption: CropOption = .scaleFill
    
    @State private var blurRadius: Double = 0.0
    @State private var sharpness: Double = 0.0
    @State private var brightness: Double = 0.0
    @State private var contrast: Double = 1.0
    
    @State private var showSettings = false
    @State private var isHoveringUpload = false
    @State private var isDropTarget = false
    @AppStorage("isMagnifierEnabled") private var isMagnifierEnabled = true // YENİ: Büyüteç Toggle Durumu
    @AppStorage("magnifierZoom") private var magnifierZoom: Double = 2.5 // YENİ: Büyüteç Zoom Seviyesi
    @State private var showHelp = false // YENİ: Kullanım Kılavuzu Durumu
    @State private var showAbout = false // YENİ: Hakkında Durumu
    
    var body: some View {
        VStack(spacing: 20) {
            
            // Header
            ZStack {
                if let logoPath = Bundle.main.path(forResource: "logo", ofType: "png"),
                   let logoImage = NSImage(contentsOfFile: logoPath) {
                    Image(nsImage: logoImage)
                        .resizable()
                        .scaledToFit()
                        .frame(height: 50)
                } else {
                    Text(loc("app_title", language))
                        .font(.largeTitle)
                        .bold()
                }
                
                HStack {
                    Spacer()
                    Button(action: { showSettings.toggle() }) {
                        Image(systemName: "line.3.horizontal")
                            .font(.system(size: 24))
                            .foregroundColor(.primary)
                    }
                    .buttonStyle(.plain)
                    .popover(isPresented: $showSettings, attachmentAnchor: .point(.bottom), arrowEdge: .top) {
                        settingsView
                    }
                }
            }
            .padding(.top, 10)
            .padding(.horizontal, 30)
            
            // Main Content Area
            HStack(spacing: 20) {
                
                // Original Image Area / Drop Zone
                VStack {
                    Text(loc("orig_image", language)).font(.headline)
                    
                    ZStack {
                        if let inputImage = inputImage {
                            Image(nsImage: inputImage)
                                .resizable()
                                .scaledToFit()
                                .cornerRadius(8)
                                .shadow(radius: 2)
                        } else {
                            Rectangle()
                                .fill(isDropTarget ? Color.accentColor.opacity(0.2) : Color.gray.opacity(0.1))
                                .cornerRadius(8)
                            
                            VStack(spacing: 15) {
                                Image(systemName: "photo.badge.plus")
                                    .font(.system(size: 40))
                                    .foregroundColor(.accentColor)
                                
                                Text(isDropTarget ? loc("drop_zone", language) : loc("click_to_select", language))
                                    .font(.headline)
                                    .foregroundColor(.accentColor)
                            }
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        selectImage()
                    }
                    .onHover { hovering in
                        isHoveringUpload = hovering
                        if hovering {
                            NSCursor.pointingHand.push()
                        } else {
                            NSCursor.pop()
                        }
                    }
                    .opacity((isHoveringUpload || isDropTarget) ? 0.8 : 1.0)
                    .animation(.easeInOut(duration: 0.2), value: isHoveringUpload)
                    .animation(.easeInOut(duration: 0.2), value: isDropTarget)
                    .onDrop(of: [UTType.image], isTargeted: $isDropTarget) { providers in
                        guard let provider = providers.first else { return false }
                        provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, error in
                            if let data = data, let image = NSImage(data: data) {
                                DispatchQueue.main.async {
                                    self.inputImage = image
                                    self.rePredict()
                                }
                            }
                        }
                        return true
                    }
                }
                
                // Result Image Area (YENİ: Büyüteç Destekli Katman)
                VStack {
                    Text(loc("depth_map", language)).font(.headline)
                    
                    ZStack {
                        if let outputImage = depthManager.outputImage {
                            // Sadece Image(nsImage: outputImage) yerine akıllı kapsayıcıyı çağırıyoruz
                            MagnifiableImageView(image: outputImage, isEnabled: isMagnifierEnabled, zoom: $magnifierZoom)
                                .cornerRadius(8)
                                .shadow(radius: 2)
                                .opacity(depthManager.isFiltering ? 0.5 : 1.0)
                        } else {
                            Rectangle()
                                .fill(Color.gray.opacity(0.1))
                                .cornerRadius(8)
                                .overlay(
                                    Group {
                                        if depthManager.isProcessing {
                                            ProgressView(loc("processing", language))
                                        } else {
                                            Text(loc("waiting", language)).foregroundColor(.gray)
                                        }
                                    }
                                )
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .padding(.horizontal, 20)
            
            // FILTERS: Main View
            GroupBox {
                HStack(spacing: 12) {
                    filterModule(
                        ModernSlider(label: loc("blur", language), value: $blurRadius, range: 0...20, step: 0.5, accentColor: .cyan, icon: "drop.fill"),
                        color: .cyan
                    )
                    .onChange(of: blurRadius) { newValue in reFilter() }
                    
                    filterModule(
                        ModernSlider(label: loc("sharpness", language), value: $sharpness, range: 0...5, step: 0.1, accentColor: .orange, icon: "sparkles"),
                        color: .orange
                    )
                    .onChange(of: sharpness) { newValue in reFilter() }
                    
                    filterModule(
                        ModernSlider(label: loc("brightness", language), value: $brightness, range: -0.5...0.5, step: 0.05, format: "%.2f", accentColor: .yellow, icon: "sun.max.fill"),
                        color: .yellow
                    )
                    .onChange(of: brightness) { newValue in reFilter() }
                    
                    filterModule(
                        ModernSlider(label: loc("contrast", language), value: $contrast, range: 0.5...2.0, step: 0.05, format: "%.2f", accentColor: .indigo, icon: "circle.lefthalf.filled"),
                        color: .indigo
                    )
                    .onChange(of: contrast) { newValue in reFilter() }
                }
                .padding(8)
            }
            .padding(.horizontal, 20)
            .opacity(depthManager.outputImage == nil ? 0.3 : 1.0)
            .disabled(depthManager.outputImage == nil)
            
            // Progress Bar
            HStack {
                if depthManager.isProcessing {
                    ProgressView(loc("neural_proc", language))
                        .progressViewStyle(.linear)
                        .frame(width: 300)
                } else {
                    Spacer().frame(height: 10)
                }
            }
            
            // Lower Buttons
            HStack {
                Spacer()
                Button(action: { saveImage() }) {
                    HStack(spacing: 10) {
                        Image(systemName: "square.and.arrow.down.fill")
                        Text(loc("save_result", language))
                            .fontWeight(.bold)
                    }
                    .padding(.horizontal, 5)
                    .frame(height: 30)
                }
                .buttonStyle(.borderedProminent)
                .tint(.blue)
                .disabled(depthManager.outputImage == nil)
                .shadow(color: .blue.opacity(0.2), radius: 5, x: 0, y: 2)
            }
            .padding(.horizontal, 30)
            .padding(.bottom, 25)
        }
        .frame(minWidth: 1250, idealWidth: 1250, minHeight: 560, idealHeight: 560)
        // YENİ: Top Menu (File -> Open / Save) Kısayol Dinleyicileri
        .onReceive(NotificationCenter.default.publisher(for: NSNotification.Name("OpenImageRequested"))) { _ in
            selectImage()
        }
        .onReceive(NotificationCenter.default.publisher(for: NSNotification.Name("SaveImageRequested"))) { _ in
            saveImage()
        }
        .onReceive(NotificationCenter.default.publisher(for: NSNotification.Name("ShowHelpRequested"))) { _ in
            showHelp = true
        }
        .sheet(isPresented: $showHelp) {
            HelpView(language: language, isPresented: $showHelp)
        }
        .onReceive(NotificationCenter.default.publisher(for: NSNotification.Name("ShowAboutRequested"))) { _ in
            showAbout = true
        }
        .sheet(isPresented: $showAbout) {
            AboutView(language: language, isPresented: $showAbout)
        }
    }
    
    // Extracted Settings Drawer
    private var settingsView: some View {
        VStack(spacing: 0) {
            // Language Picker (Centered)
            VStack(spacing: 8) {
                Text(loc("language", language))
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Picker("", selection: Binding(
                    get: { self.language },
                    set: { self.appLangStr = $0.rawValue }
                )) {
                    ForEach(AppLanguage.allCases) { option in
                        Text(option.rawValue).tag(option)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .frame(width: 160)
            }
            .padding(.top, 20)
            .padding(.bottom, 20)
            
            Divider().padding(.horizontal)
            
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Magnifier Section
                    VStack(alignment: .leading, spacing: 12) {
                        ModernToggle(label: loc("magnifier", language), isOn: $isMagnifierEnabled, icon: "plus.magnifyingglass")
                        
                        if isMagnifierEnabled {
                            ModernSlider(label: language == .tr ? "Yakınlaştırma" : "Zoom Level", 
                                       value: $magnifierZoom, 
                                       range: 1.5...5.0, 
                                       step: 0.5,
                                       accentColor: .blue,
                                       icon: "magnifyingglass")
                        }
                    }
                    
                    Divider()
                    
                    // Model Settings
                    VStack(alignment: .leading, spacing: 12) {
                        Text(loc("compute_unit", language)).font(.caption).foregroundColor(.secondary)
                        Picker("", selection: $computeUnit) {
                            ForEach(ComputeUnitOption.allCases) { option in
                                Text(option.localized(lang: language)).tag(option)
                            }
                        }
                        .labelsHidden()
                        .onChange(of: computeUnit) { newValue in
                            depthManager.loadModel(computeUnits: newValue.mlComputeUnits)
                            rePredict()
                        }
                        
                        Text(loc("scale_mode", language)).font(.caption).foregroundColor(.secondary)
                        Picker("", selection: $scaleOption) {
                            ForEach(CropOption.allCases) { option in
                                Text(option.localized(lang: language)).tag(option)
                            }
                        }
                        .labelsHidden()
                        .onChange(of: scaleOption) { _ in rePredict() }
                        
                        Text(loc("color_palette", language)).font(.caption).foregroundColor(.secondary)
                        Picker("", selection: $colorMode) {
                            ForEach(ColorMode.allCases) { option in
                                Text(option.localized(lang: language)).tag(option)
                            }
                        }
                        .pickerStyle(RadioGroupPickerStyle())
                        .labelsHidden()
                        .onChange(of: colorMode) { _ in  reFilter() }
                    }
                }
                .padding()
            }
            
                    // Footer (Centered)
                    VStack(spacing: 12) {
                        Divider().padding(.horizontal)
                        
                        VStack(spacing: 4) {
                            Text(loc("created_by", language))
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundColor(.secondary)
                            Text("© 2026")
                                .font(.system(size: 10))
                                .foregroundColor(.gray)
                        }
                        .multilineTextAlignment(.center)
                        
                        Button(action: {
                            showAbout = true
                        }) {
                            HStack(spacing: 6) {
                                Image(systemName: "info.circle")
                                Text(language == .tr ? "Hakkında" : "About")
                            }
                            .padding(.horizontal, 10)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.regular)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.bottom, 25)
                }
                .frame(width: 220, height: 500)
                .background(VisualEffectView(material: NSVisualEffectView.Material.hudWindow, blendingMode: NSVisualEffectView.BlendingMode.withinWindow))
    }
    
    // HELPERS
    private func filterModule<V: View>(_ content: V, color: Color) -> some View {
        content
            .padding(10)
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(color.opacity(0.05))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(color.opacity(0.1), lineWidth: 1)
                    )
            )
            .shadow(color: color.opacity(0.05), radius: 3, x: 0, y: 1)
    }
    
    private func selectImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [UTType.image]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        
        if panel.runModal() == .OK, let url = panel.url {
            if let image = NSImage(contentsOf: url) {
                self.inputImage = image
                rePredict()
            }
        }
    }
    
    private func rePredict() {
        guard let image = inputImage else { return }
        depthManager.predict(image: image, scaleOption: scaleOption.vnOption,
                             colorMode: colorMode, blur: blurRadius, sharpness: sharpness, brightness: brightness, contrast: contrast)
    }
    
    private func reFilter() {
        if depthManager.outputImage != nil {
            depthManager.applyFilters(colorMode: colorMode, blur: blurRadius, sharpness: sharpness, brightness: brightness, contrast: contrast)
        }
    }
    
    private func saveImage() {
        guard let image = depthManager.outputImage else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType.png]
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = "depth_output.png"
        
        if panel.runModal() == .OK, let url = panel.url {
            if let tiffData = image.tiffRepresentation,
               let bitmap = NSBitmapImageRep(data: tiffData),
               let pngData = bitmap.representation(using: .png, properties: [:]) {
                try? pngData.write(to: url)
            }
        }
    }
}

@main
struct MacApp: App {
    @AppStorage("appLang") private var appLangStr: String = AppLanguage.tr.rawValue
    private var language: AppLanguage { AppLanguage(rawValue: appLangStr) ?? .tr }

    init() {
        DispatchQueue.main.async {
            NSApp.setActivationPolicy(.regular)
            NSApp.activate(ignoringOtherApps: true)
            NSApp.windows.first?.makeKeyAndOrderFront(nil)
        }
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowStyle(HiddenTitleBarWindowStyle())
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button(loc("about_app", language)) {
                    NotificationCenter.default.post(name: NSNotification.Name("ShowAboutRequested"), object: nil)
                }
            }
            
            CommandGroup(replacing: .newItem) {
                Button(loc("file_open", language)) {
                    NotificationCenter.default.post(name: NSNotification.Name("OpenImageRequested"), object: nil)
                }
                .keyboardShortcut("o", modifiers: .command)
            }
            
            CommandGroup(replacing: .saveItem) {
                Button(loc("file_save", language)) {
                    NotificationCenter.default.post(name: NSNotification.Name("SaveImageRequested"), object: nil)
                }
                .keyboardShortcut("s", modifiers: .command)
            }
            
            CommandGroup(replacing: .help) {
                Button(loc("help_menu", language)) {
                    NotificationCenter.default.post(name: NSNotification.Name("ShowHelpRequested"), object: nil)
                }
                .keyboardShortcut("?", modifiers: .command)
            }
        }
    }
}

// MARK: - Modern UI Components
struct ModernToggle: View {
    let label: String
    @Binding var isOn: Bool
    let icon: String
    
    var body: some View {
        HStack {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(isOn ? .blue : .secondary)
                Text(label)
                    .font(.system(size: 13, weight: .medium))
            }
            
            Spacer()
            
            Button(action: { 
                withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                    isOn.toggle()
                }
            }) {
                ZStack {
                    Capsule()
                        .fill(isOn ? Color.blue.opacity(0.8) : Color.primary.opacity(0.1))
                    
                    Circle()
                        .fill(Color.white)
                        .padding(2)
                        .shadow(radius: 1)
                        .offset(x: isOn ? 10 : -10)
                }
                .frame(width: 44, height: 24)
            }
            .buttonStyle(.plain)
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.primary.opacity(0.03))
        )
    }
}

// MARK: - Helpers
struct VisualEffectView: NSViewRepresentable {
    var material: NSVisualEffectView.Material
    var blendingMode: NSVisualEffectView.BlendingMode
    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.material = material
        view.blendingMode = blendingMode
        view.state = .active
        return view
    }
    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {
        nsView.material = material
        nsView.blendingMode = blendingMode
    }
}
