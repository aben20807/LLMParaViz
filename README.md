# LLMParaViz - LLM Parallelism Visualization

An interactive Vue 3 visualization tool for understanding GPU parallelism strategies in distributed deep learning training.

## Features

- **Interactive Configuration**: Adjust TP, DP, EP, PP, CP, and FSDP sizes with sliders
- **GPU Mesh Visualization**: See how GPUs are organized in a multi-dimensional mesh
- **Detailed GPU Info**: Click any GPU to see its coordinates, data distribution, and communication groups
- **Preset Configurations**: Quick access to common parallelism setups (DDP, FSDP, Llama3-style, etc.)
- **Educational Content**: Learn about different parallelism types and their use cases

## Parallelism Types Supported

| Abbreviation | Name | Description |
|--------------|------|-------------|
| **TP** | Tensor Parallelism | Shards model layers across GPUs |
| **DP** | Data Parallelism | Replicates model, splits data batches |
| **FSDP** | Fully Sharded DP | Shards parameters, gradients, optimizer states |
| **PP** | Pipeline Parallelism | Splits model into sequential stages |
| **CP** | Context Parallelism | Shards sequence dimension |
| **EP** | Expert Parallelism | Distributes MoE experts |

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

### Build for Production

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## Deployment

This is a static site that can be deployed to any static hosting service:

- GitHub Pages
- Netlify
- Vercel
- Cloudflare Pages

After building, the `dist` folder contains all static assets.

## Inspiration

Inspired by [Visualizing 6D Mesh Parallelism](https://main-horse.github.io/posts/visualizing-6d/) by main-horse.

## License

Apache 2.0 License

## Acknowledgements

- Built with Vue 3, vite, and modern web technologies
- GitHub Copilot (Claude Opus 4.5) implemented most of the features
