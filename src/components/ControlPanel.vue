<template>
  <div class="control-panel">
    <h2>⚙️ Configuration</h2>
    
    <div class="total-gpus">
      <span class="gpu-count">{{ totalGpus }}</span>
      <span class="gpu-label">Total GPUs</span>
    </div>
    
    <div class="controls-grid">
      <div class="control-group" v-for="param in parallelParams" :key="param.key">
        <div class="control-header">
          <label :for="param.key">{{ param.label }}</label>
          <input 
            type="number" 
            :id="param.key"
            :min="param.min"
            :max="param.max"
            :value="config[param.key]"
            @input="updateConfig(param.key, $event.target.value)"
            class="number-input"
          />
        </div>
        <div class="control-description">{{ param.description }}</div>
      </div>
    </div>
    
    <div class="presets">
      <h3>Presets</h3>
      <div class="preset-buttons">
        <button 
          v-for="preset in presets" 
          :key="preset.name"
          @click="applyPreset(preset)"
          class="preset-btn"
        >
          {{ preset.name }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  config: Object,
  totalGpus: Number
})

const emit = defineEmits(['update:config'])

// Mesh order: TP → EP → CP → FSDP → DP → PP (closest to furthest)
// TP within-node (NVSwitch), EP across rail-optimized switches, CP/FSDP/DP within island, PP across islands
const parallelParams = [
  { 
    key: 'attn_tp', 
    label: 'Attention TP', 
    description: '① Closest: Within-node via NVSwitch for low-latency async TP',
    min: 1, 
    max: 64 
  },
  { 
    key: 'ulysses_dp', 
    label: 'Attention DP', 
    description: '① DP for attention only - shares TP dimension (effective_attn = TP × ADP)',
    min: 1, 
    max: 16 
  },
  { 
    key: 'expert_tp', 
    label: 'Expert TP', 
    description: '① Closest: Tensor parallelism for MoE expert FFN layers',
    min: 1, 
    max: 64 
  },
  { 
    key: 'ep', 
    label: 'EP (Expert Parallel)', 
    description: '② Rail-optimized leaf switches for best all2all perf',
    min: 1, 
    max: 256 
  },
  { 
    key: 'cp', 
    label: 'CP (Context Parallel)', 
    description: '③ Context sharding - highest latency priority in CP×FS×DP',
    min: 1, 
    max: 64 
  },
  { 
    key: 'fsdp', 
    label: 'FSDP (Fully Sharded DP)', 
    description: '④ Parameter sharding - medium latency priority',
    min: 1, 
    max: 64 
  },
  { 
    key: 'dp', 
    label: 'DP (Data Parallel)', 
    description: '⑤ Data parallel - lowest latency priority in CP×FS×DP',
    min: 1, 
    max: 64 
  },
  { 
    key: 'pp', 
    label: 'PP (Pipeline Parallel)', 
    description: '⑥ Furthest: Across islands for pipeline stages',
    min: 1, 
    max: 64 
  },
  { 
    key: 'gpusPerNode', 
    label: 'GPUs per Node', 
    description: 'Number of GPUs per physical node (for grouping)',
    min: 1, 
    max: 16 
  }
]

const presets = [
  { name: 'Single GPU', config: { attn_tp: 1, expert_tp: 1, ulysses_dp: 1, dp: 1, ep: 1, pp: 1, cp: 1, fsdp: 1, gpusPerNode: 8 }},
  { name: '8x DDP', config: { attn_tp: 1, expert_tp: 1, ulysses_dp: 1, dp: 8, ep: 1, pp: 1, cp: 1, fsdp: 1, gpusPerNode: 8 }},
  { name: '8x FSDP', config: { attn_tp: 1, expert_tp: 1, ulysses_dp: 1, dp: 1, ep: 1, pp: 1, cp: 1, fsdp: 8, gpusPerNode: 8 }},
  { name: '2D (TP+DP)', config: { attn_tp: 2, expert_tp: 2, ulysses_dp: 1, dp: 4, ep: 1, pp: 1, cp: 1, fsdp: 1, gpusPerNode: 8 }},
  { name: '3D (TP+DP+PP)', config: { attn_tp: 2, expert_tp: 2, ulysses_dp: 1, dp: 2, ep: 1, pp: 2, cp: 1, fsdp: 1, gpusPerNode: 8 }},
  { name: 'Llama3-style', config: { attn_tp: 8, expert_tp: 8, ulysses_dp: 1, dp: 1, ep: 1, pp: 4, cp: 2, fsdp: 2, gpusPerNode: 8 }},
  { name: 'MoE (EP only)', config: { attn_tp: 1, expert_tp: 1, ulysses_dp: 1, dp: 2, ep: 8, pp: 1, cp: 1, fsdp: 1, gpusPerNode: 8 }},
  { name: 'MoE + TP', config: { attn_tp: 4, expert_tp: 2, ulysses_dp: 1, dp: 2, ep: 4, pp: 1, cp: 1, fsdp: 1, gpusPerNode: 8 }},
  { name: 'DeepSeek-style', config: { attn_tp: 4, expert_tp: 1, ulysses_dp: 1, dp: 2, ep: 8, pp: 1, cp: 1, fsdp: 1, gpusPerNode: 8 }},
  { name: 'Attention DP', config: { attn_tp: 8, expert_tp: 16, ulysses_dp: 2, dp: 1, ep: 1, pp: 1, cp: 1, fsdp: 1, gpusPerNode: 8 }},
  { name: 'ATP8+ADP2+EP8', config: { attn_tp: 8, expert_tp: 1, ulysses_dp: 2, dp: 1, ep: 8, pp: 1, cp: 1, fsdp: 1, gpusPerNode: 8 }},
  { name: '64 GPU Full', config: { attn_tp: 2, expert_tp: 2, ulysses_dp: 1, dp: 2, ep: 2, pp: 2, cp: 2, fsdp: 2, gpusPerNode: 8 }}
]

const updateConfig = (key, value) => {
  emit('update:config', {
    ...props.config,
    [key]: parseInt(value)
  })
}

const applyPreset = (preset) => {
  emit('update:config', { ...preset.config })
}
</script>

<style scoped>
.control-panel {
  background: var(--panel-bg);
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.control-panel h2 {
  margin: 0 0 20px 0;
  font-size: var(--font-xl);
  font-weight: 700;
  color: var(--text-primary);
}

.total-gpus {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
  background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
  border-radius: 10px;
  margin-bottom: 28px;
}

.gpu-count {
  font-size: 3.5rem;
  font-weight: 800;
  color: white;
  line-height: 1;
}

.gpu-label {
  color: rgba(255, 255, 255, 0.95);
  font-size: var(--font-base);
  font-weight: 500;
  margin-top: 6px;
}

.controls-grid {
  display: grid;
  gap: 16px;
}

.control-group {
  background: var(--control-bg);
  padding: 16px;
  border-radius: 10px;
}

.control-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
}

.control-header label {
  font-weight: 600;
  color: var(--text-primary);
  font-size: var(--font-sm);
}

.number-input {
  width: 64px;
  padding: 10px 12px;
  border: 2px solid var(--border-color);
  border-radius: 8px;
  background: var(--panel-bg);
  color: var(--text-primary);
  font-size: var(--font-base);
  font-weight: 700;
  text-align: center;
  appearance: textfield;
  -moz-appearance: textfield;
  transition: border-color 0.15s, box-shadow 0.15s;
}

.number-input:hover {
  border-color: var(--text-secondary);
}

.number-input:focus {
  outline: none;
  border-color: var(--primary-color);
  box-shadow: var(--focus-ring);
}

.number-input::-webkit-inner-spin-button,
.number-input::-webkit-outer-spin-button {
  -webkit-appearance: none;
  margin: 0;
}

.control-description {
  font-size: var(--font-xs);
  color: var(--text-secondary);
  line-height: var(--leading-relaxed);
  margin-top: 4px;
}

.presets {
  margin-top: 28px;
  padding-top: 24px;
  border-top: 1px solid var(--border-color);
}

.presets h3 {
  margin: 0 0 14px 0;
  font-size: var(--font-sm);
  font-weight: 600;
  color: var(--text-secondary);
}

.preset-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.preset-btn {
  padding: 8px 14px;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  background: var(--control-bg);
  color: var(--text-primary);
  font-size: var(--font-xs);
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;
}

.preset-btn:hover {
  background: var(--primary-color);
  color: white;
  border-color: var(--primary-color);
}

.preset-btn:focus-visible {
  box-shadow: var(--focus-ring);
}
</style>
