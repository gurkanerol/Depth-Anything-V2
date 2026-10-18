import Foundation
import CoreML

let packageURL = URL(fileURLWithPath: "DepthAnythingV2SmallF16.mlpackage")
do {
    let compiledURL = try MLModel.compileModel(at: packageURL)
    print("COMPILED TO: \(compiledURL.path)")
    
    // Copy to our current directory for easy access
    let destURL = URL(fileURLWithPath: "DepthAnythingV2SmallF16.mlmodelc")
    if FileManager.default.fileExists(atPath: destURL.path) {
        try FileManager.default.removeItem(at: destURL)
    }
    try FileManager.default.copyItem(at: compiledURL, to: destURL)
    print("MOVED TO: \(destURL.path)")
} catch {
    print("ERROR: \(error)")
}
