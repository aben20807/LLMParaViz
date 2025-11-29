<template>
  <div class="app">
    <header class="header">
      <h1>🔲 LLMParaViz</h1>
      <p class="subtitle">Interactive LLM Parallelism Visualization for Distributed Inference/Training</p>
    </header>
    
    <main class="main-content">
      <ControlPanel 
        v-model:config="parallelConfig"
        :total-gpus="totalGPUs"
      />
      
      <div class="visualization-area">
        <GPUGrid 
          :config="parallelConfig"
          :selected-gpu="selectedGPU"
          @select-gpu="toggleGPU"
        />
        
        <GPUDetails 
          v-if="selectedGPU !== null"
          :gpu-id="selectedGPU"
          :config="parallelConfig"
        />
        
        <CommunicationView :config="parallelConfig" />
      </div>
      
      <InfoPanel :config="parallelConfig" />
    </main>
    
    <footer class="footer">
      <p class="footer-credit">
        Created by <strong>Huang, Po-Hsuan (aben20807)</strong> · 
        <a href="https://github.com/aben20807/LLMParaViz" target="_blank">GitHub</a> · 
        Inspired by <a href="https://main-horse.github.io/posts/visualizing-6d/" target="_blank">Visualizing 6D Mesh Parallelism</a>
      </p>
      <p class="footer-disclaimer">
        For educational purposes only. Actual implementations may vary. Contributions welcome!
      </p>
    </footer>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import ControlPanel from './components/ControlPanel.vue'
import GPUGrid from './components/GPUGrid.vue'
import GPUDetails from './components/GPUDetails.vue'
import InfoPanel from './components/InfoPanel.vue'
import CommunicationView from './components/CommunicationView.vue'

const parallelConfig = ref({
  attn_tp: 2,  // Attention Tensor Parallelism
  expert_tp: 2, // Expert Tensor Parallelism (for MoE)
  ulysses_dp: 1, // Attention DP (DP for attention layer only, shares TP dimension)
  dp: 2,  // Data Parallelism
  ep: 1,  // Expert Parallelism
  pp: 1,  // Pipeline Parallelism
  cp: 1,  // Context Parallelism
  fsdp: 1, // Fully Sharded Data Parallelism
  gpusPerNode: 8 // GPUs per node (for visualization grouping)
})

const selectedGPU = ref(null)

function toggleGPU(gpuId) {
  selectedGPU.value = selectedGPU.value === gpuId ? null : gpuId
}

const totalGPUs = computed(() => {
  const c = parallelConfig.value
  // Attention TP and EP can share the same GPU dimension (orthogonal sharding)
  // Attention DP also shares with attention TP: effective_attn_tp = attn_tp * ulysses_dp
  // Expert TP further shards each expert, so it multiplies
  // Formula: max(attn_tp * ulysses_dp, ep) × expert_tp × cp × fsdp × dp × pp
  const effective_attn = c.attn_tp * c.ulysses_dp
  const base = Math.max(effective_attn, c.ep)
  return base * c.expert_tp * c.cp * c.fsdp * c.dp * c.pp
})
</script>
