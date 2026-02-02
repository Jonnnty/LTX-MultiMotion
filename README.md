# LTX-MultiMotion: Number-free Person Motion Generation from Text
Built on LTX-Video-2B's lightweight architecture, LTX-MultiMotion re-engineers encoder and decoder to transform the spatial width dimension into a number-of-person-related representation. This enables dynamic motion generation for varying character counts without pre-defined constraints.

<div align="center">
  <img src="assets/demo.jpg" width="400" alt="项目示意图">
</div>

## Key Innovations
- **Person-Aware Architecture**: We repurpose the spatial width dimension to dynamically represent varying numbers of characters in a unified latent space.
- **Specialized Pathways**: A triple-branch decoder with dedicated networks isolates and generates translation, orientation, and pose components in parallel.
- **Progressive Expansion**: The encoder and decoder network begins with a shallow structure and dynamically increases its depth layer-by-layer during training, adapting its capacity to the complexity of the motion data.
