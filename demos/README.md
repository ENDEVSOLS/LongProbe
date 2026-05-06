# 🎬 LongProbe Demo Recordings

This directory contains scripts to generate animated terminal demos for LongProbe.

## 🛠️ Setup

Install VHS (by Charm):

```bash
# macOS
brew install vhs

# Linux
sudo snap install vhs

# Or with Go
go install github.com/charmbracelet/vhs@latest
```

## 📹 Generate Demos

```bash
# Generate all demos
make demos

# Or generate individually
vhs demos/01-quick-start.tape
vhs demos/02-detect-regression.tape
vhs demos/03-pytest-integration.tape
vhs demos/04-generate-questions.tape
```

## 📦 Output

Generated files will be in `demos/` directory:
- `01-quick-start.gif` - Quick start guide
- `02-detect-regression.gif` - Regression detection
- `03-pytest-integration.gif` - Pytest integration
- `04-generate-questions.gif` - Auto-generate questions

## 🎨 Customization

Edit `.tape` files to customize:
- `Set Theme` - Color scheme (Dracula, Monokai, etc.)
- `Set FontSize` - Text size
- `Set Width/Height` - Dimensions
- `Set TypingSpeed` - Animation speed
- `Output` - Output format (gif, mp4, webm)

## 📚 VHS Documentation

- [VHS GitHub](https://github.com/charmbracelet/vhs)
- [VHS Examples](https://github.com/charmbracelet/vhs/tree/main/examples)

## 🔄 Updating Demos

When LongProbe features change:
1. Edit the corresponding `.tape` script
2. Regenerate: `vhs demos/XX-demo-name.tape`
3. Commit both `.tape` and `.gif` files

## 📝 Demo Scripts

### 01-quick-start.tape
Shows basic installation and first test run.

### 02-detect-regression.tape
Demonstrates regression detection with baseline comparison.

### 03-pytest-integration.tape
Shows how to integrate LongProbe with pytest.

### 04-generate-questions.tape
Demonstrates auto-generating golden questions from documents.

## 🎯 Alternative: asciinema

If you prefer asciinema (lighter weight):

```bash
# Install
pip install asciinema

# Record
asciinema rec demo.cast

# Upload (gets shareable link)
asciinema upload demo.cast

# Embed in README
[![asciicast](https://asciinema.org/a/XXXXX.svg)](https://asciinema.org/a/XXXXX)
```

## 📊 File Sizes

GIFs can be large. Consider:
- **GIF**: Easy to embed, larger file size (~2-5MB)
- **MP4**: Smaller, requires video player (~500KB)
- **WebM**: Smallest, best quality (~300KB)
- **SVG**: Smallest, but limited browser support

To generate MP4 instead of GIF:

```bash
# Change in .tape file
Output demos/01-quick-start.mp4
```

## 🚀 GitHub README Integration

Add to README.md:

```markdown
## 🎬 Quick Demo

![LongProbe Demo](demos/01-quick-start.gif)

<details>
<summary>More Demos</summary>

### Regression Detection
![Regression Detection](demos/02-detect-regression.gif)

### Pytest Integration
![Pytest Integration](demos/03-pytest-integration.gif)

### Auto-Generate Questions
![Generate Questions](demos/04-generate-questions.gif)

</details>
```
