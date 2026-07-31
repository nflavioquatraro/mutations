import AppKit

let canvasWidth = 1200
let canvasHeight = 630
let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let publicDirectory = root.appendingPathComponent("public")
let markURL = publicDirectory.appendingPathComponent("logo/mutations-mark.png")
let wordmarkURL = publicDirectory.appendingPathComponent("logo/mutations-wordmark.png")
let outputURL = publicDirectory.appendingPathComponent("og.png")

func color(_ red: CGFloat, _ green: CGFloat, _ blue: CGFloat, alpha: CGFloat = 1) -> NSColor {
    NSColor(srgbRed: red / 255, green: green / 255, blue: blue / 255, alpha: alpha)
}

guard
    let mark = NSImage(contentsOf: markURL),
    let wordmark = NSImage(contentsOf: wordmarkURL),
    let bitmap = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: canvasWidth,
        pixelsHigh: canvasHeight,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bitmapFormat: [],
        bytesPerRow: 0,
        bitsPerPixel: 0
    ),
    let graphics = NSGraphicsContext(bitmapImageRep: bitmap)
else {
    fatalError("Unable to create the Open Graph image context or load the logo assets.")
}

NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = graphics
graphics.imageInterpolation = .high

let canvas = NSRect(x: 0, y: 0, width: canvasWidth, height: canvasHeight)
color(248, 250, 252).setFill()
canvas.fill()

let grid = color(28, 80, 126, alpha: 0.07)
grid.setStroke()
let gridPath = NSBezierPath()
gridPath.lineWidth = 1

for x in stride(from: 0, through: canvasWidth, by: 120) {
    gridPath.move(to: NSPoint(x: x, y: 0))
    gridPath.line(to: NSPoint(x: x, y: canvasHeight))
}

for y in stride(from: 0, through: canvasHeight, by: 105) {
    gridPath.move(to: NSPoint(x: 0, y: y))
    gridPath.line(to: NSPoint(x: canvasWidth, y: y))
}

gridPath.stroke()

let circlePath = NSBezierPath(ovalIn: NSRect(x: 608, y: 56, width: 520, height: 520))
circlePath.lineWidth = 1
color(48, 119, 182, alpha: 0.16).setStroke()
circlePath.stroke()

color(90, 168, 230, alpha: 0.46).setFill()
for x: CGFloat in [238, 478, 718, 958] {
    NSBezierPath(ovalIn: NSRect(x: x - 2.5, y: 523.5, width: 5, height: 5)).fill()
}

NSGraphicsContext.saveGraphicsState()
graphics.cgContext.setShadow(
    offset: CGSize(width: 0, height: -12),
    blur: 30,
    color: color(18, 62, 105, alpha: 0.08).cgColor
)
mark.draw(in: NSRect(x: 92, y: 285, width: 196, height: 203))
NSGraphicsContext.restoreGraphicsState()

wordmark.draw(in: NSRect(x: 346, y: 336, width: 515, height: 89))

let subtitleAttributes: [NSAttributedString.Key: Any] = [
    .font: NSFont(name: "Helvetica Neue", size: 34) ?? NSFont.systemFont(ofSize: 34),
    .foregroundColor: color(22, 63, 104),
    .kern: -0.68,
]

NSAttributedString(
    string: "A framework for modular, routed adaptations around",
    attributes: subtitleAttributes
).draw(at: NSPoint(x: 92, y: 194))

NSAttributedString(
    string: "frozen language models.",
    attributes: subtitleAttributes
).draw(at: NSPoint(x: 92, y: 150))

color(31, 91, 145).setFill()
NSRect(x: 92, y: 54, width: 140, height: 2).fill()

NSGraphicsContext.restoreGraphicsState()

guard let png = bitmap.representation(using: .png, properties: [:]) else {
    fatalError("Unable to encode the Open Graph image as PNG.")
}

try png.write(to: outputURL)
print("Generated \(outputURL.path) (\(canvasWidth)x\(canvasHeight))")
