<template>
  <div class="gpu-grid-container">
    <h2>🖥️ GPU Mesh Visualization</h2>
    
    <div class="mesh-info">
      <span class="mesh-dim" v-for="dim in activeDimensions" :key="dim.key">
        <span class="dim-label" :style="{ color: dim.color }">{{ dim.label }}</span>
        <span class="dim-value">{{ dim.value }}</span>
      </span>
      <span class="mesh-dim node-info">
        <span class="dim-label">Nodes</span>
        <span class="dim-value">{{ numNodes }}</span>
      </span>
      <span class="mesh-dim node-info">
        <span class="dim-label">GPUs/Node</span>
        <span class="dim-value">{{ config.gpusPerNode }}</span>
      </span>
    </div>
    
    <div class="grid-wrapper">
      <div class="nodes-container">
        <div 
          v-for="node in nodesList" 
          :key="node.id" 
          class="node-group"
          :style="getNodeStyle(node.id)"
        >
          <div class="node-header">Node {{ node.id }}</div>
          <div class="node-grid" :style="getNodeGridStyle(node)">
            <div 
              v-for="gpu in node.gpus" 
              :key="gpu.id"
              class="gpu-cell"
              :class="{ 
                selected: gpu.id === selectedGpu,
                highlighted: isHighlighted(gpu)
              }"
              @click="$emit('select-gpu', gpu.id)"
              @mouseenter="hoveredGpu = gpu.id"
              @mouseleave="hoveredGpu = null"
            >
              <div class="gpu-header">
                <div class="gpu-id">{{ gpu.id }}</div>
              </div>
              <div class="gpu-labels">
                <span 
                  v-for="label in gpu.labels" 
                  :key="label.key"
                  class="gpu-label"
                  :style="{ background: label.color }"
                  :title="label.title"
                >
                  {{ label.text }}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <div class="legend">
      <h4>Legend</h4>
      <div class="legend-items">
        <div 
          v-for="dim in activeDimensions" 
          :key="dim.key"
          class="legend-item"
        >
          <span class="legend-color" :style="{ background: dim.color }"></span>
          <span class="legend-label">{{ dim.fullLabel }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  config: Object,
  selectedGpu: Number
})

const emit = defineEmits(['select-gpu'])

const hoveredGpu = ref(null)

const dimensionColors = {
  attn_tp: '#ef4444',   // Red - Attention Tensor Parallel
  ulysses_dp: '#dc2626', // Dark Red - Attention DP
  expert_tp: '#f97316', // Orange - Expert Tensor Parallel
  dp: '#3b82f6',   // Blue - Data Parallel
  ep: '#10b981',   // Green - Expert Parallel
  pp: '#f59e0b',   // Amber - Pipeline Parallel
  cp: '#8b5cf6',   // Purple - Context Parallel
  fsdp: '#ec4899'  // Pink - FSDP
}

const dimensionLabels = {
  attn_tp: { short: 'ATP', full: 'Attention TP' },
  ulysses_dp: { short: 'ADP', full: 'Attention DP' },
  expert_tp: { short: 'ETP', full: 'Expert TP' },
  dp: { short: 'DP', full: 'Data Parallel' },
  ep: { short: 'EP', full: 'Expert Parallel' },
  pp: { short: 'PP', full: 'Pipeline Parallel' },
  cp: { short: 'CP', full: 'Context Parallel' },
  fsdp: { short: 'FS', full: 'FSDP' }
}

const activeDimensions = computed(() => {
  const dims = []
  // Mesh order: TP → Ulysses → EP → CP → FSDP → DP → PP (closest to furthest)
  const order = ['attn_tp', 'ulysses_dp', 'expert_tp', 'ep', 'cp', 'fsdp', 'dp', 'pp']
  
  for (const key of order) {
    if (props.config[key] > 1) {
      dims.push({
        key,
        label: dimensionLabels[key].short,
        fullLabel: dimensionLabels[key].full,
        value: props.config[key],
        color: dimensionColors[key]
      })
    }
  }
  return dims
})

const totalGPUs = computed(() => {
  const c = props.config
  // Attention TP and EP can share the same GPU dimension (orthogonal sharding)
  // Ulysses DP (attention DP) shares with attention TP: effective_attn = attn_tp * ulysses_dp
  // Expert TP further shards each expert, so it multiplies
  // Formula: max(attn_tp * ulysses_dp, ep) × expert_tp × cp × fsdp × dp × pp
  const effective_attn = c.attn_tp * (c.ulysses_dp || 1)
  const base = Math.max(effective_attn, c.ep)
  return base * c.expert_tp * c.cp * c.fsdp * c.dp * c.pp
})

const numNodes = computed(() => {
  return Math.ceil(totalGPUs.value / props.config.gpusPerNode)
})

const gridDimensions = computed(() => {
  const total = totalGPUs.value
  // Try to make a reasonable grid
  const sqrt = Math.sqrt(total)
  let cols = Math.ceil(sqrt)
  
  // Prefer certain column counts for common GPU setups
  if (total % 8 === 0 && total >= 8) cols = 8
  else if (total % 4 === 0 && total >= 4) cols = Math.min(8, total)
  else if (total % 2 === 0) cols = Math.min(8, total)
  
  const rows = Math.ceil(total / cols)
  return { rows, cols }
})

const gridStyle = computed(() => {
  const { rows, cols } = gridDimensions.value
  return {
    gridTemplateColumns: `repeat(${cols}, 1fr)`,
    gridTemplateRows: `repeat(${rows}, 1fr)`
  }
})

const gpuList = computed(() => {
  const gpus = []
  const c = props.config
  const total = totalGPUs.value
  
  for (let i = 0; i < total; i++) {
    const coords = getGPUCoordinates(i)
    const groups = getGPUGroups(i, coords)
    const labels = getGPULabels(coords)
    const nodeId = Math.floor(i / c.gpusPerNode)
    
    gpus.push({
      id: i,
      coords,
      groups,
      labels,
      nodeId
    })
  }
  return gpus
})

const nodesList = computed(() => {
  const nodes = []
  const gpusPerNode = props.config.gpusPerNode
  
  for (let nodeId = 0; nodeId < numNodes.value; nodeId++) {
    const nodeGpus = gpuList.value.filter(gpu => gpu.nodeId === nodeId)
    nodes.push({
      id: nodeId,
      gpus: nodeGpus
    })
  }
  return nodes
})

function getNodeStyle(nodeId) {
  const nodeHue = (nodeId * 50) % 360
  return {
    borderColor: `hsl(${nodeHue}, 50%, 50%)`
  }
}

function getNodeGridStyle(node) {
  const gpuCount = node.gpus.length
  // Prefer 4 columns for typical 8 GPU nodes (2x4 layout)
  let cols = 4
  if (gpuCount <= 4) {
    cols = gpuCount
  } else if (gpuCount <= 8) {
    cols = 4
  } else if (gpuCount <= 16) {
    cols = 8
  } else {
    cols = Math.min(8, Math.ceil(gpuCount / 2))
  }
  return {
    gridTemplateColumns: `repeat(${cols}, 1fr)`
  }
}

function getGPUCoordinates(gpuId) {
  const c = props.config
  // Decompose GPU ID into coordinates for each parallelism dimension
  // Ulysses DP shares with attention TP: effective_attn = attn_tp * ulysses_dp
  // Base dimension: max(effective_attn, ep) - they share GPUs (orthogonal sharding)
  // Then: expert_tp -> cp -> fsdp -> dp -> pp
  const ulysses = c.ulysses_dp || 1
  const effective_attn = c.attn_tp * ulysses
  const base = Math.max(effective_attn, c.ep)
  let remaining = gpuId
  
  // Base coordinate (shared by attn_tp, ulysses_dp, and ep)
  const baseCoord = remaining % base
  remaining = Math.floor(remaining / base)
  
  // Attention TP and Ulysses DP coordinates derived from base
  // attn_tp is innermost, ulysses_dp wraps around it
  const attn_tp = baseCoord % c.attn_tp
  const ulysses_dp = Math.floor(baseCoord / c.attn_tp) % ulysses
  
  // EP coordinate derived from base
  const ep = baseCoord % c.ep
  
  // Expert TP is a separate dimension that multiplies
  const expert_tp = remaining % c.expert_tp
  remaining = Math.floor(remaining / c.expert_tp)
  
  const cp = remaining % c.cp
  remaining = Math.floor(remaining / c.cp)
  
  const fsdp = remaining % c.fsdp
  remaining = Math.floor(remaining / c.fsdp)
  
  const dp = remaining % c.dp
  remaining = Math.floor(remaining / c.dp)
  
  const pp = remaining % c.pp
  
  return { baseCoord, attn_tp, ulysses_dp, expert_tp, ep, cp, fsdp, dp, pp }
}

function getGPUGroups(gpuId, coords) {
  const c = props.config
  const ulysses = c.ulysses_dp || 1
  const effective_attn = c.attn_tp * ulysses
  const base = Math.max(effective_attn, c.ep)
  // Calculate which group this GPU belongs to for each parallelism type
  
  // Attention TP group: GPUs that share attention computation
  // Same ulysses_dp, expert_tp, cp, fsdp, dp, pp, but different attn_tp coord
  const attnTpGroup = coords.ulysses_dp +
                      coords.expert_tp * ulysses + 
                      coords.cp * ulysses * c.expert_tp + 
                      coords.fsdp * ulysses * c.expert_tp * c.cp +
                      coords.dp * ulysses * c.expert_tp * c.cp * c.fsdp +
                      coords.pp * ulysses * c.expert_tp * c.cp * c.fsdp * c.dp
  
  // Ulysses DP group: GPUs with same attn_tp but different ulysses_dp (attention replicas)
  const ulyssesDpGroup = coords.attn_tp +
                         coords.expert_tp * c.attn_tp + 
                         coords.cp * c.attn_tp * c.expert_tp + 
                         coords.fsdp * c.attn_tp * c.expert_tp * c.cp +
                         coords.dp * c.attn_tp * c.expert_tp * c.cp * c.fsdp +
                         coords.pp * c.attn_tp * c.expert_tp * c.cp * c.fsdp * c.dp
  
  // EP group: GPUs that do All2All for expert routing
  const epGroup = coords.expert_tp + 
                  coords.cp * c.expert_tp + 
                  coords.fsdp * c.expert_tp * c.cp +
                  coords.dp * c.expert_tp * c.cp * c.fsdp +
                  coords.pp * c.expert_tp * c.cp * c.fsdp * c.dp
  
  // Expert TP group: GPUs that shard same expert
  const expertTpGroup = coords.baseCoord + 
                        coords.cp * base + 
                        coords.fsdp * base * c.cp +
                        coords.dp * base * c.cp * c.fsdp +
                        coords.pp * base * c.cp * c.fsdp * c.dp
  
  // DP group
  const dpGroup = coords.baseCoord + 
                  coords.expert_tp * base + 
                  coords.cp * base * c.expert_tp +
                  coords.fsdp * base * c.expert_tp * c.cp +
                  coords.pp * base * c.expert_tp * c.cp * c.fsdp
  
  // PP group
  const ppGroup = coords.baseCoord + 
                  coords.expert_tp * base + 
                  coords.cp * base * c.expert_tp +
                  coords.fsdp * base * c.expert_tp * c.cp +
                  coords.dp * base * c.expert_tp * c.cp * c.fsdp
  
  return { attnTp: attnTpGroup, ulyssesDp: ulyssesDpGroup, expertTp: expertTpGroup, ep: epGroup, dp: dpGroup, pp: ppGroup }
}

function getGPULabels(coords) {
  const labels = []
  const c = props.config
  
  // Labels ordered by mesh hierarchy: TP → Ulysses → EP → CP → FSDP → DP → PP
  if (c.attn_tp > 1) {
    labels.push({
      key: 'attn_tp',
      text: `A${coords.attn_tp}`,
      color: dimensionColors.attn_tp,
      title: `Attention TP Rank ${coords.attn_tp}`
    })
  }
  
  if ((c.ulysses_dp || 1) > 1) {
    labels.push({
      key: 'ulysses_dp',
      text: `AD${coords.ulysses_dp}`,
      color: dimensionColors.ulysses_dp,
      title: `Attention DP Rank ${coords.ulysses_dp}`
    })
  }
  
  if (c.expert_tp > 1) {
    labels.push({
      key: 'expert_tp',
      text: `X${coords.expert_tp}`,
      color: dimensionColors.expert_tp,
      title: `Expert TP Rank ${coords.expert_tp}`
    })
  }
  
  if (c.ep > 1) {
    labels.push({
      key: 'ep',
      text: `E${coords.ep}`,
      color: dimensionColors.ep,
      title: `Expert Parallel Rank ${coords.ep}`
    })
  }
  
  if (c.cp > 1) {
    labels.push({
      key: 'cp',
      text: `C${coords.cp}`,
      color: dimensionColors.cp,
      title: `Context Parallel Rank ${coords.cp}`
    })
  }
  
  if (c.fsdp > 1) {
    labels.push({
      key: 'fsdp',
      text: `F${coords.fsdp}`,
      color: dimensionColors.fsdp,
      title: `FSDP Rank ${coords.fsdp}`
    })
  }
  
  if (c.dp > 1) {
    labels.push({
      key: 'dp',
      text: `D${coords.dp}`,
      color: dimensionColors.dp,
      title: `Data Parallel Rank ${coords.dp}`
    })
  }
  
  if (c.pp > 1) {
    labels.push({
      key: 'pp',
      text: `P${coords.pp}`,
      color: dimensionColors.pp,
      title: `Pipeline Stage ${coords.pp}`
    })
  }
  
  return labels
}

function isHighlighted(gpu) {
  if (hoveredGpu.value === null) return false
  if (hoveredGpu.value === gpu.id) return true
  
  // Highlight GPUs in the same TP group
  const hoveredGpuData = gpuList.value[hoveredGpu.value]
  if (hoveredGpuData && hoveredGpuData.groups.attnTp === gpu.groups.attnTp) {
    return true
  }
  
  return false
}
</script>

<style scoped>
.gpu-grid-container {
  background: var(--panel-bg);
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.gpu-grid-container h2 {
  margin: 0 0 20px 0;
  font-size: var(--font-xl);
  font-weight: 700;
  color: var(--text-primary);
}

.mesh-info {
  display: flex;
  gap: 16px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}

.mesh-dim {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: var(--font-sm);
}

.dim-label {
  font-weight: 600;
  color: var(--text-primary);
}

.dim-value {
  background: var(--control-bg);
  padding: 4px 10px;
  border-radius: 6px;
  font-weight: 700;
  color: var(--text-primary);
}

.grid-wrapper {
  overflow-x: auto;
  padding: 12px;
  background: var(--control-bg);
  border-radius: 10px;
}

.nodes-container {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
}

.node-group {
  background: var(--panel-bg);
  border: 3px solid var(--border-color);
  border-radius: 12px;
  padding: 14px;
  min-width: fit-content;
}

.node-header {
  font-size: var(--font-sm);
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 12px;
  text-align: center;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border-color);
}

.node-grid {
  display: grid;
  gap: 10px;
}

.grid-container {
  display: grid;
  gap: 10px;
  min-width: fit-content;
}

.gpu-cell {
  position: relative;
  min-width: 76px;
  min-height: 76px;
  background: var(--control-bg);
  border: 2px solid var(--border-color);
  border-radius: 10px;
  padding: 10px;
  cursor: pointer;
  transition: all 0.15s;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.gpu-cell:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  border-color: var(--text-secondary);
}

.gpu-cell:focus-visible {
  outline: none;
  box-shadow: var(--focus-ring);
}

.gpu-cell.selected {
  border-color: #818cf8;
  box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.4);
}

.gpu-cell.highlighted {
  background: rgba(129, 140, 248, 0.15);
}

.gpu-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}

.gpu-id {
  font-size: var(--font-xl);
  font-weight: 800;
  color: var(--text-primary);
}

.gpu-node {
  font-size: var(--font-xs);
  padding: 2px 6px;
  border-radius: 4px;
  background: var(--border-color);
  color: var(--text-primary);
  font-weight: 600;
}

.node-info .dim-label {
  color: var(--text-secondary);
}

.gpu-labels {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: auto;
}

.gpu-label {
  font-size: var(--font-xs);
  padding: 3px 6px;
  border-radius: 4px;
  color: #ffffff;
  font-weight: 700;
  text-shadow: 0 1px 1px rgba(0,0,0,0.3);
}

.legend {
  margin-top: 24px;
  padding-top: 20px;
  border-top: 1px solid var(--border-color);
}

.legend h4 {
  margin: 0 0 14px 0;
  font-size: var(--font-sm);
  font-weight: 600;
  color: var(--text-secondary);
}

.legend-items {
  display: flex;
  flex-wrap: wrap;
  gap: 18px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: var(--font-sm);
  color: var(--text-primary);
}

.legend-color {
  width: 14px;
  height: 14px;
  border-radius: 4px;
}
</style>
